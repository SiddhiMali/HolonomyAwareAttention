"""
checkpoint_baselines.py -- dense-label checkpoint span experiment, all baselines, params matched
===============================================================================================
Same task, labels, gates, energy bins and TEST SETS as checkpoint_span.py --dense. Two
differences, both needed for a fair comparison with position-encoded models:

1. VARIABLE-LENGTH TRAINING. In checkpoint_span.py every training sequence had length 17,
   so the QUERY always sat at position 16 and the checkpoint's position alone revealed the
   span; standard attention learned absolute-position rules that broke at test. Here each
   training sequence has prefix P ~ U[0, train_prefix_max] and span S ~ U[1, 12], so the
   total length varies (2..27 by default). Batches are grouped by length (no padding).
   Test sequences (prefix 8, spans 4..48) are identical to checkpoint_span.py.
   -> spans 4..12 are in-distribution in span AND position; 16 is longer in span only;
      24+ are longer in both.

2. PARAMETER MATCHING. Target = holo's parameter count. For each baseline, the size knob
   that exists is searched to get closest to the target:
     standard (sinusoidal PE) : number of layers        (checkpoint_compare_kernel / v5)
     rope                     : number of layers        (implemented here: pre-norm blocks)
     sung                     : recurrent state size    (v5)
     mamba, gru               : model width d_model     (v5; embedding/head scale with it)
   Every arm prints its parameter count and the chosen setting.

Dense supervision for every arm: label after every gate since the most recent checkpoint.
Validation and test read only the QUERY, as before.

RUN (Colab, same folder as checkpoint_span.py, checkpoint_compare_kernel.py, qgate_*.py,
     attention/utils files)
  python checkpoint_baselines.py --selftest
  python checkpoint_baselines.py --arms holo_es1.5 standard rope sung mamba gru --epochs 60
"""
import argparse, copy, math
import numpy as np
from checkpoint_span4 import (make_span_sample, make_span_dataset, setup, dense_labels,
                             get_args as _span_args)
from qgate_checkpoint import vocab_size, _rollout_energy, CKPT_ID, QUERY_ID
from qgate_energy import bin_energy


# =====================================================================
# DATA (numpy)
# =====================================================================
def make_var_dataset(n, args, gates_c, H, edges, seed):
    rng = np.random.default_rng(seed)
    seqs, Y, E, Ed, Yd, Md = [], [], [], [], [], []
    for _ in range(n):
        S = int(rng.integers(1, args.train_span_max + 1))
        P = int(rng.integers(0, args.train_prefix_max + 1))
        seq, span = make_span_sample(P, S, args.K, args.p_ckpt, rng)
        e = _rollout_energy(span, gates_c, H)
        ed, yd, md = dense_labels(seq[None, :], args.K, gates_c, H, edges)
        seqs.append(seq); Y.append(bin_energy(e, edges)); E.append(e)
        Ed.append(ed[0]); Yd.append(yd[0]); Md.append(md[0])
    return dict(seqs=seqs, Y=np.array(Y, dtype=np.int64), E=np.array(E, dtype=np.float32),
                Ed=Ed, Yd=Yd, Md=Md)

def buckets(idx, seqs):
    out = {}
    for i in idx:
        out.setdefault(len(seqs[i]), []).append(i)
    return out

def selftest(args):
    gates_c, H, edges = setup(args)
    print("=" * 72); print("CHECKPOINT BASELINES SELF-TEST (numpy)"); print("=" * 72)
    d = make_var_dataset(3000, args, gates_c, H, edges, 1)
    lens = np.array([len(s) for s in d["seqs"]])
    print(f"training lengths {lens.min()}-{lens.max()} (mean {lens.mean():.1f}); "
          f"{len(buckets(range(len(lens)), d['seqs']))} length buckets")
    qpos = lens - 1
    print(f"QUERY position varies: {qpos.min()}-{qpos.max()} (was always 16 in checkpoint_span.py)")
    ok = all(abs(d["Ed"][i][-1] - d["E"][i]) < 1e-5 for i in range(len(lens)))
    print(f"[ok] dense QUERY label equals the final label for all {len(lens)} sequences" if ok else "FAIL")
    assert ok
    print(f"     labelled positions per sequence: {np.mean([m.sum() for m in d['Md']]):.1f}")
    spans_ok = True
    for S in args.test_spans:
        X, Y, E = make_span_dataset(50, args.K, args.p_ckpt, gates_c, H, edges, 1000 + S,
                                    S=S, P=args.test_prefix)
        spans_ok &= X.shape[1] == args.test_prefix + S + 2
    print(f"[ok] test sets built exactly as in checkpoint_span.py (prefix {args.test_prefix}, "
          f"spans {args.test_spans}; seeds 1000+S)" if spans_ok else "FAIL")
    print(f"in-distribution positions: training reaches length {lens.max()}; test length at span 12 = "
          f"{args.test_prefix + 14}, at span 16 = {args.test_prefix + 18}, at span 24 = {args.test_prefix + 26}")


# =====================================================================
# TORCH
# =====================================================================
def run(args):
    import torch, torch.nn as nn
    from checkpoint_compare_kernel3 import QueryReadoutModel, evaluate, to_tensors, DEVICE
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gates_c, H, edges = setup(args)
    vocab = vocab_size(args.K)

    # ---------------- RoPE baseline (same interface as QueryReadoutModel) ----------------
    def rope(x):                                          # x: (B, h, L, dk)
        L, dk = x.shape[-2], x.shape[-1]; half = dk // 2
        freqs = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / half))
        ang = torch.arange(L, device=x.device).float()[:, None] * freqs[None, :]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    class RopeLayer(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.h, self.dk = h, d // h
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        def forward(self, x):
            B, L, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
            q, k = rope(q), rope(k)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            att = att.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1),
                                  float("-inf")).softmax(-1)
            x = x + self.out((att @ v).transpose(1, 2).reshape(B, L, D))
            return x + self.ff(self.ln2(x))

    class RopeInner(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            d = args.d_model
            self.embed = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([RopeLayer(d, args.n_heads) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d)
            self.cls_head = nn.Linear(d, args.n_bins); self.reg_head = nn.Linear(d, 1)
        def encode(self, x):
            h = self.embed(x)
            for l in self.layers:
                h = l(h)
            return self.ln_f(h)

    class RopeReadout(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.inner = RopeInner(n_layers); self.curved = False
        def forward(self, x):
            h = self.inner.encode(x)[:, -1]
            return self.inner.cls_head(h), self.inner.reg_head(h).squeeze(-1)

    def build(kind, **over):
        if kind == "rope":
            return RopeReadout(over.get("n_layers", 4))
        kw = dict(d_model=args.d_model, n_heads=args.n_heads, n_std_layers=args.n_std_layers,
                  edge_scale=args.edge_scale, sung_n_state=args.sung_n_state,
                  sung_scale=args.sung_scale, theta=args.theta)
        kw.update(over)
        return QueryReadoutModel(kind, vocab, args.n_bins, **kw)

    count = lambda m: sum(p.numel() for p in m.parameters())

    # ---------------- parameter matching ----------------
    target = count(build(args.match_to))
    print(f"parameter target = {args.match_to}: {target:,}")
    def match(kind):
        if kind.startswith("holo"):
            return {}, count(build(kind))
        grids = {"standard": ("n_std_layers", list(range(1, 25))),
                 "rope": ("n_layers", list(range(1, 25))),
                 "sung": ("sung_n_state", [16, 24, 32, 48, 64, 80, 96, 128, 160, 192]),
                 "mamba": ("d_model", [32, 48, 64, 80, 96, 128, 160, 192, 256]),
                 "gru": ("d_model", [32, 48, 64, 80, 96, 128, 160, 192, 256])}
        if kind not in grids:
            return {}, count(build(kind))
        key, vals = grids[kind]; best = None
        for v in vals:
            try:
                c = count(build(kind, **{key: v}))
            except Exception:
                continue
            if best is None or abs(c - target) < abs(best[1] - target):
                best = ({key: v}, c)
        return best

    # ---------------- data ----------------
    tr = make_var_dataset(args.n_train, args, gates_c, H, edges, args.seed)
    n = len(tr["seqs"]); nval = n // 10
    val_b = buckets(range(nval), tr["seqs"]); tr_b = buckets(range(nval, n), tr["seqs"])
    tests = {}
    for S in args.test_spans:
        X, Y, E = make_span_dataset(args.n_test, args.K, args.p_ckpt, gates_c, H, edges,
                                    1000 + S, S=S, P=args.test_prefix)
        tests[S] = (to_tensors(X, Y, E), float(np.abs(E - np.median(E)).mean()))
    T = lambda a: torch.from_numpy(np.stack(a))

    @torch.no_grad()
    def val_acc(model):
        model.eval(); c = t = 0
        for L, ids in val_b.items():
            for i in range(0, len(ids), args.bs):
                b = ids[i:i + args.bs]
                logits, _ = model(T([tr["seqs"][j] for j in b]).to(DEVICE))
                c += (logits.argmax(-1).cpu().numpy() == tr["Y"][b]).sum(); t += len(b)
        return c / t

    per_arm_lr = {"holo": args.holo_lr, "gru": args.gru_lr, "mamba": args.mamba_lr,
                  "standard": args.std_lr, "sung": args.sung_lr, "rope": args.std_lr}

    res, info = {}, {}
    for kind in args.arms:
        over, nparam = match(kind)
        model = build(kind, **over).to(DEVICE)
        info[kind] = (nparam, over)
        print(f"\n--- {kind} --- params {nparam:,} ({100 * nparam / target:.0f}% of target) {over or ''}")
        lr = per_arm_lr.get("holo" if kind.startswith("holo") else kind) or args.lr
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
        best, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            batches = [ids[i:i + args.bs] for ids in
                       (list(np.random.permutation(v)) for v in tr_b.values())
                       for i in range(0, len(ids), args.bs)]
            np.random.shuffle(batches)
            for b in batches:
                xb = T([tr["seqs"][j] for j in b]).to(DEVICE)
                m = T([tr["Md"][j] for j in b]).to(DEVICE)
                opt.zero_grad()
                h = model.inner.encode(xb)
                logits = model.inner.cls_head(h)[m]
                epred = model.inner.reg_head(h).squeeze(-1)[m]
                loss = ce(logits, T([tr["Yd"][j] for j in b]).to(DEVICE)[m]) + \
                       args.reg_weight * mse(epred, T([tr["Ed"][j] for j in b]).to(DEVICE)[m])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            acc = val_acc(model)
            if acc > best:
                best, best_state = acc, copy.deepcopy(model.state_dict())
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [{kind}] ep {epoch + 1}/{args.epochs} loss={loss.item():.3f} val_acc={acc:.3f} best={best:.3f}")
        model.load_state_dict(best_state)
        res[kind] = {S: evaluate(model, *tests[S][0], args.n_bins) for S in args.test_spans}
        for S in args.test_spans:
            print(f"  {kind} span={S:>3} acc={res[kind][S][0]:.3f}  mae={res[kind][S][1]:.4f}")

    print("\n" + "=" * 76)
    print("PARAMETERS: " + ", ".join(f"{k} {info[k][0]:,}" for k in args.arms))
    print("\nMAE vs SPAN (lower is better)  -- dense labels, variable-length training")
    print(f"{'span':>5} {'const':>8} " + " ".join(f"{k:>11}" for k in args.arms))
    for S in args.test_spans:
        print(f"{S:>5} {tests[S][1]:>8.4f} " + " ".join(f"{res[k][S][1]:>11.4f}" for k in args.arms))
    print("\nACCURACY vs SPAN")
    print(f"{'span':>5} " + " ".join(f"{k:>11}" for k in args.arms))
    for S in args.test_spans:
        print(f"{S:>5} " + " ".join(f"{res[k][S][0]:>11.3f}" for k in args.arms))


def get_args():
    import sys
    ap_extra = argparse.ArgumentParser(add_help=False)
    ap_extra.add_argument("--train-prefix-max", dest="train_prefix_max", type=int, default=14)
    ap_extra.add_argument("--match-to", dest="match_to", default="holo_es1.5")
    extra, rest = ap_extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = _span_args()                                   # all checkpoint_span.py options
    if args.arms == ["holo", "standard", "sung", "gru"]:  # default arms for this file
        args.arms = ["holo_es1.5", "standard", "rope", "sung", "mamba", "gru"]
    args.train_prefix_max, args.match_to = extra.train_prefix_max, extra.match_to
    return args


if __name__ == "__main__":
    a = get_args()
    selftest(a) if a.selftest else run(a)
