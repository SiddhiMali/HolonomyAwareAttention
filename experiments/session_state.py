"""
session_state.py -- session memory, REDESIGNED for Smriti (no questions, dense turn labels)
==========================================================================================
The memory reads only the session. Nothing in the stream asks anything.

STREAM   same sentences as session_memory.py: sets, relative edits, per-setting reverts,
         "never mind , go back ." (contextual undo), markers, filler, "ok , keep it ."
STEPS    every token rotates the state (no pooling / chunking)
LABELS   at every turn boundary ('.'), four probe heads predict the FULL current state:
         timeout (10 classes), retries (10), auth (3), cache (2). Dense supervision at turn
         level: after filler turns the label is UNCHANGED, so drift is penalized right there.
FILLER   training histories sometimes contain a quiet run of 0-16 pure-filler sentences
         inserted at a random point.

EVALUATION (distribution shifts the labels never covered)
  state at the end of: events 12 (in-distribution), 24, 48; and after quiet stretches
  (12 events + 16 / 48 filler sentences at the end)
  accuracy by depth into long histories (turns 1-16, 17-32, 33+ of the 48-event test)
  baselines on the same question: last-write-wins, retrieval-5 (compose 5 most similar)

ARMS   holo<L>:<edge_scale>, standard (auto-matched to the first holo arm by #layers, or
       standard:<L>), gru (auto-matched by hidden size). Same labels for every arm.
MEASURE for holo arms: mean rotation size on tokens of filler turns vs state-changing turns
        (does it learn to rotate only when something changes?)

RUN
  python session_state.py --selftest
  python session_state.py --arms holo1:0.05 holo3:0.01 standard gru --epochs 30
"""
import argparse, copy, importlib, math
import numpy as np
import session_memory as sm

SETTINGS, DOMAIN = sm.SETTINGS, sm.DOMAIN
N_CLS = {s: len(DOMAIN[s]) for s in SETTINGS}
PURE_FILLERS = sm.FILLERS[:4]

def cls_of(s, v):
    return DOMAIN[s].index(v)

# =====================================================================
# DATA
# =====================================================================
def replay_states(sents):
    """exact state after every sentence (independent replay via session_memory._apply)"""
    stack = {s: [] for s in SETTINGS}; log = []; out = []
    for i, sent in enumerate(sents):
        if sm._marker_of(sent) is None:
            if i < 4:
                w = sent.split(); s = "auth" if w[0] == "switch" else w[2]; stack[s].append(w[-2])
            else:
                sm._apply(sent, stack, log)
        out.append({s: (stack[s][-1] if stack[s] else None) for s in SETTINGS})   # None = not set yet
    return out

def sentence_kind(sent, i):
    if i < 4: return "init"
    if sm._marker_of(sent): return "marker"
    if sent in sm.FILLERS: return "filler"
    return "change"

def make_history_ex(rng, n_events, d_quiet_end=0, quiet_train=False):
    sents, stack, snaps = sm.make_history(rng, n_events, d_quiet=d_quiet_end)
    if quiet_train and rng.random() < 0.5:                       # quiet run inside the history
        k = int(rng.integers(0, 17)); pos = int(rng.integers(5, len(sents) + 1))
        sents = sents[:pos] + [PURE_FILLERS[rng.integers(4)] for _ in range(k)] + sents[pos:]
    return sents

def encode(sents):
    toks, lab, kind_tok, turn_of_tok = [], [], [], []
    states = replay_states(sents)
    for i, (sent, st) in enumerate(zip(sents, states)):
        kd = sentence_kind(sent, i)
        for w in sent.split():
            toks.append(sm.W2I[w]); kind_tok.append(kd); turn_of_tok.append(i)
            lab.append([(cls_of(s, st[s]) if st[s] is not None else -1) for s in SETTINGS]
                       if w == "." else [-1] * 4)
    return np.array(toks), np.array(lab), kind_tok, np.array(turn_of_tok)

def make_set(n, seed, ev_lo, ev_hi, d_quiet_end=0, quiet_train=False):
    rng = np.random.default_rng(seed)
    hs = [make_history_ex(rng, int(rng.integers(ev_lo, ev_hi + 1)), d_quiet_end, quiet_train) for _ in range(n)]
    enc = [encode(h) for h in hs]
    return hs, enc

def pad(enc):
    L = max(len(e[0]) for e in enc)
    X = np.zeros((len(enc), L), dtype=np.int64); Y = -np.ones((len(enc), L, 4), dtype=np.int64)
    lens = np.zeros(len(enc), dtype=np.int64)
    for i, (t, lab, _, _) in enumerate(enc):
        X[i, :len(t)] = t; Y[i, :len(t)] = lab; lens[i] = len(t)
    return X, Y, lens

# baselines: final state per setting
def base_final(sents, which, k=5):
    out = {}
    for s in SETTINGS:
        q = f"? what is the {s} now"
        out[s] = sm.baseline_lww(sents, "current", q) if which == "lww" else sm.baseline_retrieval(sents, "current", q, k)
    return out

# =====================================================================
# SELFTEST
# =====================================================================
def selftest(args):
    print("=" * 76); print("SESSION STATE (redesign) SELF-TEST -- no questions, labels at every turn"); print("=" * 76)
    rng = np.random.default_rng(3)
    h = make_history_ex(rng, 8, quiet_train=True)
    t, lab, kd, turn = encode(h)
    print("sample:", " ".join(h))
    st = replay_states(h)
    print(f"  state after last turn: {st[-1]}")
    print(f"  tokens {len(t)}, labelled turn ends {int((lab.max(1) >= 0).sum())}")
    # independent check: labels at turn ends equal a from-scratch simulate() on the prefix
    hs, enc = make_set(400, 11, 8, 16, quiet_train=True)
    bad = 0
    for sents, (tk, lb, _, trn) in zip(hs, enc):
        ends = np.where(lb.max(1) >= 0)[0]
        for e in ends[::3]:
            i = trn[e]; stack, _ = sm.simulate(sents[:i + 1])
            want = [(cls_of(s, stack[s][-1]) if stack[s] else -1) for s in SETTINGS]
            bad += list(lb[e]) != want
    print(f"[ok] labels match an independent replay at sampled turn ends ({'all' if bad == 0 else bad} checked)")
    assert bad == 0
    print(f"\n{'config':<18}{'tokens':>10}{'turns':>7} | {'majority':>9} {'LWW':>6} {'ret5':>6}   (final-state accuracy, mean of 4 settings)")
    for name, lo, hi, dq, qt in (("train (8-16,quiet)", 8, 16, 0, True), ("events 12", 12, 12, 0, False),
                                 ("events 24", 24, 24, 0, False), ("events 48", 48, 48, 0, False),
                                 ("quiet 12+D16", 12, 12, 16, False), ("quiet 12+D48", 12, 12, 48, False)):
        hs, enc = make_set(600, 7, lo, hi, dq, qt)
        lens = np.array([len(e[0]) for e in enc]); turns = np.array([len(x) for x in hs])
        finals = [replay_states(x)[-1] for x in hs]
        maj = np.mean([max(np.bincount([cls_of(s, f[s]) for f in finals]) / len(finals)) for s in SETTINGS])
        lww = np.mean([np.mean([base_final(x, "lww")[s] == f[s] for x, f in zip(hs, finals)]) for s in SETTINGS])
        ret = np.mean([np.mean([base_final(x, "ret")[s] == f[s] for x, f in zip(hs, finals)]) for s in SETTINGS])
        print(f"{name:<18}{f'{lens.min()}-{lens.max()}':>10}{int(turns.mean()):>7} | {maj:>9.2f} {lww:>6.2f} {ret:>6.2f}")

# =====================================================================
# TORCH
# =====================================================================
def run(args):
    import torch, torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    MHA = importlib.import_module(args.attn_module).PathOrderedGeometricMHA
    from utils import skew_from_vec
    d = args.d_model

    def sinusoidal_pe(L, dm, device):
        pe = torch.zeros(L, dm, device=device); pos = torch.arange(L, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dm, 2, device=device).float() * (-math.log(10000.0) / dm))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); return pe

    class HoloBlock(nn.Module):
        def __init__(self, es):
            super().__init__()
            self.core = MHA(d, n_heads=args.n_heads, edge_scale=es, mode="causal")
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        def forward(self, h, mask):
            h = h + self.core(self.ln1(h), attn_mask=mask, return_geometry=False)[0]
            return h + self.ff(self.ln2(h))

    class Model(nn.Module):
        def __init__(self, arm, std_layers=None, gru_hidden=None):
            super().__init__()
            self.kind = arm.split(":")[0].rstrip("0123456789") if arm.startswith("holo") else arm.split(":")[0]
            self.embed = nn.Embedding(sm.VOCAB, d)
            width = gru_hidden or d
            self.heads = nn.ModuleList([nn.Linear(width, N_CLS[s]) for s in SETTINGS])
            if self.kind == "standard":
                layer = nn.TransformerEncoderLayer(d, args.n_heads, 4 * d, batch_first=True, dropout=0.0)
                self.core = nn.TransformerEncoder(layer, std_layers)
            elif self.kind == "gru":
                self.core = nn.GRU(d, gru_hidden, batch_first=True)
            else:
                lay, es = arm[len("holo"):].split(":"); self.n = int(lay); es = float(es)
                if self.n == 1:
                    self.core = MHA(d, n_heads=args.n_heads, edge_scale=es, mode="causal")
                else:
                    self.blocks = nn.ModuleList([HoloBlock(es) for _ in range(self.n)])
        def hidden(self, x):
            h = self.embed(x); L = h.shape[1]
            if self.kind == "standard":
                h = h + sinusoidal_pe(L, d, x.device).unsqueeze(0)
                return self.core(h, mask=torch.triu(torch.ones(L, L, device=x.device), 1).bool())
            if self.kind == "gru":
                return self.core(h)[0]
            mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
            if self.n == 1:
                return h + self.core(h, attn_mask=mask, return_geometry=False)[0]
            for b in self.blocks:
                h = b(h, mask)
            return h
        def forward(self, x):
            h = self.hidden(x)
            return [hd(h) for hd in self.heads]
        def first_core_input(self, x):
            e = self.embed(x)
            return (self.core, e) if self.n == 1 else (self.blocks[0].core, self.blocks[0].ln1(e))

    count = lambda m: sum(p.numel() for p in m.parameters())
    holo_arms = [a for a in args.arms if a.startswith("holo")]
    target = count(Model(holo_arms[0])) if holo_arms else None
    def build(arm):
        if arm.startswith("standard"):
            if ":" in arm:
                return Model(arm, std_layers=int(arm.split(":")[1]))
            best = min(range(1, 25), key=lambda n: abs(count(Model("standard", std_layers=n)) - target))
            return Model("standard", std_layers=best)
        if arm == "gru":
            best = min([32, 48, 64, 96, 128, 160, 192, 256, 320], key=lambda w: abs(count(Model("gru", gru_hidden=w)) - target))
            return Model("gru", gru_hidden=best)
        return Model(arm)

    # ---------------- data ----------------
    hs_tr, enc_tr = make_set(args.n_hist, args.seed + 100, 8, 16, 0, quiet_train=True)
    nv = len(enc_tr) // 10
    Xv, Yv, lv = pad(enc_tr[:nv]); tr = enc_tr[nv:]
    configs = {"events 12": (12, 12, 0), "events 24": (24, 24, 0), "events 48": (48, 48, 0),
               "quiet 12+D16": (12, 12, 16), "quiet 12+D48": (12, 12, 48)}
    tests = {}
    for i, (name, (lo, hi, dq)) in enumerate(configs.items()):
        hs, enc = make_set(args.n_test_hist, 900 + i, lo, hi, dq)
        tests[name] = (hs, enc, pad(enc))
    t = lambda z: torch.from_numpy(z).to(DEVICE)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    @torch.no_grad()
    def predict(m, X, bs=32):
        m.eval(); outs = [[] for _ in SETTINGS]
        for i in range(0, len(X), bs):
            for k, lg in enumerate(m(t(X[i:i + bs]))):
                outs[k].append(lg.argmax(-1).cpu().numpy())
        return [np.concatenate(o) for o in outs]                    # 4 x (N, L)

    def turn_acc(pred, Y):
        return float(np.mean([(pred[k][Y[..., k] >= 0] == Y[..., k][Y[..., k] >= 0]).mean() for k in range(4)]))

    res = {}
    for arm in args.arms:
        m = build(arm).to(DEVICE)
        print(f"\n--- {arm} --- params {count(m):,}" + (f" ({100 * count(m) / target:.0f}% of target)" if target else ""))
        opt = torch.optim.Adam(m.parameters(), lr=args.lr); best, best_state = -1, None
        for ep in range(args.epochs):
            m.train(); perm = np.random.permutation(len(tr))
            for i in range(0, len(perm), args.bs):
                X, Y, _ = pad([tr[j] for j in perm[i:i + args.bs]])
                opt.zero_grad(); logits = m(t(X)); Yt = t(Y)
                loss = sum(ce(lg.reshape(-1, lg.shape[-1]), Yt[..., k].reshape(-1)) for k, lg in enumerate(logits))
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            vacc = turn_acc(predict(m, Xv), Yv)
            if vacc > best:
                best, best_state = vacc, copy.deepcopy(m.state_dict())
            if ep == 0 or (ep + 1) % 5 == 0:
                print(f"  ep {ep + 1}/{args.epochs} loss={loss.item():.3f} val(turn-state acc)={vacc:.3f} best={best:.3f}")
        m.load_state_dict(best_state)
        r = {}
        for name, (hs, enc, (X, Y, lens)) in tests.items():
            pred = predict(m, X)
            fin = np.mean([np.mean([pred[k][i, lens[i] - 1] == Y[i, lens[i] - 1, k] for i in range(len(X))]) for k in range(4)])
            r[name] = {"final": fin, "all_turns": turn_acc(pred, Y)}
            if name == "events 48":
                depth = {}
                for lo_t, hi_t, lab_ in ((0, 20, "turns 1-20"), (20, 36, "turns 21-36"), (36, 999, "turns 37+")):
                    acc = []
                    for i, (_, lb, _, trn) in enumerate(enc):
                        ends = np.where((lb.max(1) >= 0) & (trn >= lo_t) & (trn < hi_t))[0]
                        for k in range(4):
                            ok_ = Y[i, ends, k] >= 0
                            acc.extend(pred[k][i, ends][ok_] == Y[i, ends, k][ok_])
                    depth[lab_] = float(np.mean(acc)) if acc else float("nan")
                r["depth"] = depth
        if arm.startswith("holo"):                                   # rotation size by turn type
            with torch.no_grad():
                hs, enc, (X, Y, lens) = tests["events 12"]
                core, e_in = m.first_core_input(t(X[:32])); B, L, _ = e_in.shape
                pair = torch.cat([e_in, torch.roll(e_in, shifts=-1, dims=1)], dim=-1)
                a = core.edge_mlp(pair).view(B, L, core.h, core.m_dim).transpose(1, 2) * core.edge_scale
                size = skew_from_vec(a, core.d).flatten(-2).norm(dim=-1).mean(1).cpu().numpy()   # (B, L) edge t->t+1
                by = {"filler": [], "change": [], "marker": []}
                for i in range(B):
                    kd = enc[i][2]
                    for p in range(lens[i] - 1):
                        if kd[p + 1] in by:
                            by[kd[p + 1]].append(size[i, p])
                r["rot"] = {k: float(np.mean(v)) for k, v in by.items() if v}
            print(f"  rotation size by turn type: " + ", ".join(f"{k} {v:.3f}" for k, v in r["rot"].items()))
        res[arm] = r

    print("\n" + "=" * 76); print("FINAL-STATE ACCURACY (mean of the 4 settings)  -- baselines on the same question")
    names = list(tests)
    print("arm".ljust(18) + "".join(n.rjust(14) for n in names))
    for bname in ("lww", "ret"):
        row = []
        for n in names:
            hs = tests[n][0]; finals = [replay_states(x)[-1] for x in hs]
            row.append(np.mean([np.mean([base_final(x, bname)[s] == f[s] for x, f in zip(hs, finals)]) for s in SETTINGS]))
        print(("last-write-wins" if bname == "lww" else "retrieval-5").ljust(18) + "".join(f"{v:.3f}".rjust(14) for v in row))
    for arm, r in res.items():
        print(arm.ljust(18) + "".join(f"{r[n]['final']:.3f}".rjust(14) for n in names))
    print("\nACCURACY AT EVERY TURN (state after each turn, mean of 4 settings)")
    for arm, r in res.items():
        print(arm.ljust(18) + "".join(f"{r[n]['all_turns']:.3f}".rjust(14) for n in names))
    print("\nDEPTH into 48-event histories (state accuracy at turn ends)")
    for arm, r in res.items():
        print(arm.ljust(18) + "  ".join(f"{k}: {v:.3f}" for k, v in r["depth"].items()))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--arms", nargs="+", default=["holo1:0.05", "holo3:0.01", "standard", "gru"])
    p.add_argument("--attn-module", dest="attn_module", default="attention_fast_chunked")
    p.add_argument("--n-hist", dest="n_hist", type=int, default=20000)
    p.add_argument("--n-test-hist", dest="n_test_hist", type=int, default=400)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", dest="d_model", type=int, default=64)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    selftest(a) if a.selftest else run(a)
