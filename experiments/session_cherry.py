"""
session_cherry.py -- CHERRY-PICK, phase 1 (no stored geometry). Session-state family.
====================================================================================
INPUT = the session only. No questions anywhere in the stream.

STATE     timeout (0-9), retries (0-9), auth (sessions / jwt / oauth), cache (on / off)
SESSION   [4 initial sets] [segment] marker v1 [segment] marker v2 ... marker vN [segment]
          markers : "we just shipped release vK ."       (K = 1..N, in order)
          segments: REVERSIBLE relative edits only (a group action on the state):
              increase / decrease the timeout by k .   increase / decrease the retries by k .
              toggle the cache .      cycle the auth method .  (sessions -> jwt -> oauth -> ...)
              swap the timeout and retries .           <- makes order matter
          light filler ("running the tests .") at a low rate. No sets after the start.

DENSE LABELS  at every turn end ('.'): the full current state, read by 4 stream heads.

CHERRY-PICK PROBE (the host's call; its arguments never enter the stream)
    apply(from=A, to=B, start=X)  ->  the 4 settings obtained by applying, in order, every
    edit between marker A and marker B to the starting state X.
    Probe (identical for every model, deliberately small):
        query = MLP( embed(word "vA"), embed(word "vB"), embed(X) )   (marker words share the
                                                                        model's token embeddings)
        ONE cross-attention layer over the model's hidden states at every stream position
        4 probe heads (timeout, retries, auth, cache)
    Several probes per session (different A, B, X) read the same hidden states, so no history
    is duplicated.

TEST AXES   in-distribution (2-6 edits/segment, 3-4 markers) | longer spans (6-12, 3-4)
            | more markers (2-6, 6-8) | both (6-12, 6-8)
BASELINES   unchanged (answer = X) | order-blind (same edits, order ignored) |
            retrieval-5 (5 statements most similar to the call, composed perfectly)
METRICS     probe exact match (all 4 right), probe per-setting accuracy, stream turn accuracy

PREDICTIONS (written before running)
    GRU degrades as markers multiply; standard degrades as spans lengthen; holo uncertain in
    phase 1 -- a hidden state at a marker is a STATE, and two states do not determine the
    transformation between them, so success needs the memory to carry transformations itself.

RUN
    python session_cherry.py --selftest
    python session_cherry.py --arms holo3:0.05 standard gru --epochs 30
"""
import argparse, copy, importlib, math
import numpy as np

SETTINGS = ["timeout", "retries", "auth", "cache"]
AUTHS = ["sessions", "jwt", "oauth"]
CACHES = ["on", "off"]
N_CLS = [10, 10, 3, 2]
MAX_MARK = 8
FILLERS = ["running the tests .", "looks good .", "let me check the logs .", "tests are passing ."]

_words = set(". set the to switch auth increase decrease by timeout retries toggle cache cycle method "
             "swap and we just shipped release running tests looks good let me check logs are passing on off".split())
_words |= set(AUTHS) | {str(i) for i in range(10)} | {f"v{k}" for k in range(1, MAX_MARK + 1)}
WORDS = ["<pad>"] + sorted(_words)
W2I = {w: i for i, w in enumerate(WORDS)}
VOCAB = len(WORDS)

# =====================================================================
# GENERATOR (structured records -> sentences)
# =====================================================================
def _op_sentence(rng):
    r = rng.random()
    if r < 0.40:
        s = "timeout" if rng.random() < 0.5 else "retries"; k = int(rng.integers(1, 4))
        d = "increase" if rng.random() < 0.5 else "decrease"
        return f"{d} the {s} by {k} ."
    if r < 0.55: return "toggle the cache ."
    if r < 0.70: return "cycle the auth method ."
    return "swap the timeout and retries ."

def _apply_gen(sent, st):                 # generator's own implementation
    w = sent.split()
    if w[0] in ("increase", "decrease"):
        i = 0 if w[2] == "timeout" else 1; k = int(w[4])
        st[i] = (st[i] + (k if w[0] == "increase" else -k)) % 10
    elif w[0] == "toggle": st[3] = 1 - st[3]
    elif w[0] == "cycle": st[2] = (st[2] + 1) % 3
    elif w[0] == "swap": st[0], st[1] = st[1], st[0]

def make_session(rng, ops_lo, ops_hi, mk_lo, mk_hi, p_filler=0.15):
    st = [int(rng.integers(10)), int(rng.integers(10)), int(rng.integers(3)), int(rng.integers(2))]
    init = [f"set the timeout to {st[0]} .", f"set the retries to {st[1]} .",
            f"switch auth to {AUTHS[st[2]]} .", f"set the cache to {CACHES[st[3]]} ."]
    order = rng.permutation(4); sents = [init[i] for i in order]
    n_mark = int(rng.integers(mk_lo, mk_hi + 1)); mark_idx = {}
    for seg in range(n_mark + 1):
        for _ in range(int(rng.integers(ops_lo, ops_hi + 1))):
            sents.append(_op_sentence(rng))
            if rng.random() < p_filler:
                sents.append(FILLERS[rng.integers(4)])
        if seg < n_mark:
            mark_idx[seg + 1] = len(sents); sents.append(f"we just shipped release v{seg + 1} .")
    return sents, mark_idx

def span_ops(sents, mark_idx, A, B):
    return [s for s in sents[mark_idx[A] + 1: mark_idx[B]] if s.split()[0] in
            ("increase", "decrease", "toggle", "cycle", "swap")]

def gen_states(sents):
    """stream states via the GENERATOR's implementation (initial sets parsed, then _apply_gen)"""
    st = [None] * 4; out = []
    for s_ in sents:
        w = s_.split()
        if w[0] == "set":
            i = {"timeout": 0, "retries": 1, "cache": 3}[w[2]]
            st[i] = CACHES.index(w[4]) if i == 3 else int(w[4])
        elif w[0] == "switch":
            st[2] = AUTHS.index(w[3])
        elif None not in st:
            _apply_gen(s_, st)
        out.append(list(st))
    return out

def cherry_gen(sents, mark_idx, A, B, X):
    st = list(X)
    for s in span_ops(sents, mark_idx, A, B):
        _apply_gen(s, st)
    return st

def make_probes(rng, sents, mark_idx, n_probes):
    N = len(mark_idx); out = []
    for _ in range(n_probes):
        A, B = sorted(rng.choice(np.arange(1, N + 1), size=2, replace=False).tolist())
        X = [int(rng.integers(10)), int(rng.integers(10)), int(rng.integers(3)), int(rng.integers(2))]
        out.append((A, B, X, cherry_gen(sents, mark_idx, A, B, X)))
    return out

# =====================================================================
# INDEPENDENT VERIFICATION (re-parses the text; different code path)
# =====================================================================
def replay_states(sents):
    t = r = a = c = None; out = []
    for s in sents:
        w = s.split()
        if w[0] == "set" and w[2] == "timeout": t = int(w[4])
        elif w[0] == "set" and w[2] == "retries": r = int(w[4])
        elif w[0] == "set" and w[2] == "cache": c = CACHES.index(w[4])
        elif w[0] == "switch": a = AUTHS.index(w[3])
        elif w[0] == "increase" and w[2] == "timeout": t = (t + int(w[4])) % 10
        elif w[0] == "decrease" and w[2] == "timeout": t = (t - int(w[4])) % 10
        elif w[0] == "increase" and w[2] == "retries": r = (r + int(w[4])) % 10
        elif w[0] == "decrease" and w[2] == "retries": r = (r - int(w[4])) % 10
        elif w[0] == "toggle": c = 1 - c
        elif w[0] == "cycle": a = (a + 1) % 3
        elif w[0] == "swap": t, r = r, t
        out.append([t, r, a, c])
    return out

def cherry_verify(sents, A, B, X):
    ia = sents.index(f"we just shipped release v{A} ."); ib = sents.index(f"we just shipped release v{B} .")
    t, r, a, c = X
    for s in sents[ia + 1:ib]:
        w = s.split()
        if w[0] == "increase" and w[2] == "timeout": t = (t + int(w[4])) % 10
        elif w[0] == "decrease" and w[2] == "timeout": t = (t - int(w[4])) % 10
        elif w[0] == "increase" and w[2] == "retries": r = (r + int(w[4])) % 10
        elif w[0] == "decrease" and w[2] == "retries": r = (r - int(w[4])) % 10
        elif w[0] == "toggle": c = 1 - c
        elif w[0] == "cycle": a = (a + 1) % 3
        elif w[0] == "swap": t, r = r, t
    return [t, r, a, c]

# =====================================================================
# BASELINES
# =====================================================================
def base_unchanged(sents, mark_idx, A, B, X):
    return list(X)

def base_order_blind(sents, mark_idx, A, B, X):
    dt = dr = tog = cyc = sw = 0
    for s in span_ops(sents, mark_idx, A, B):
        w = s.split()
        if w[0] in ("increase", "decrease"):
            k = int(w[4]) * (1 if w[0] == "increase" else -1)
            if w[2] == "timeout": dt += k
            else: dr += k
        elif w[0] == "toggle": tog += 1
        elif w[0] == "cycle": cyc += 1
        elif w[0] == "swap": sw += 1
    t, r = (X[0] + dt) % 10, (X[1] + dr) % 10
    if sw % 2: t, r = r, t
    return [t, r, (X[2] + cyc) % 3, (X[3] + tog) % 2]

def base_retrieval(sents, mark_idx, A, B, X, k=5):
    q = {f"v{A}", f"v{B}", "release", "timeout", "retries", "auth", "cache"}
    stop = {"the", ".", "by", "to", "and", "we", "just", "shipped"}
    scored = sorted(((len((set(s.split()) - stop) & q), i) for i, s in enumerate(sents)), key=lambda z: (-z[0], -z[1]))
    keep = {i for _, i in scored[:k]}
    st = list(X)
    for i in range(mark_idx[A] + 1, mark_idx[B]):           # generous: true span boundaries
        if i in keep:
            _apply_gen(sents[i], st)
    return st

BASELINES = {"unchanged": base_unchanged, "order-blind": base_order_blind, "retrieval-5": base_retrieval}

# =====================================================================
# DATASETS
# =====================================================================
CONFIGS = {"in-dist": (2, 6, 3, 4), "longer spans": (6, 12, 3, 4),
           "more markers": (2, 6, 6, 8), "both": (6, 12, 6, 8)}

def build(n, seed, cfg, n_probes):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        sents, mark_idx = make_session(rng, *cfg)
        rows.append((sents, mark_idx, make_probes(rng, sents, mark_idx, n_probes)))
    return rows

def encode(rows):
    toks, labs = [], []
    for sents, _, _ in rows:
        states = replay_states(sents); t, l = [], []
        for s, st in zip(sents, states):
            for w in s.split():
                t.append(W2I[w]); l.append([(v if v is not None else -1) for v in st] if w == "." else [-1] * 4)
        toks.append(t); labs.append(l)
    L = max(len(t) for t in toks)
    X = np.zeros((len(rows), L), dtype=np.int64); Y = -np.ones((len(rows), L, 4), dtype=np.int64)
    for i, (t, l) in enumerate(zip(toks, labs)):
        X[i, :len(t)] = t; Y[i, :len(t)] = l
    P = len(rows[0][2])
    PA = np.array([[p[0] for p in r[2]] for r in rows]); PB = np.array([[p[1] for p in r[2]] for r in rows])
    PX = np.array([[p[2] for p in r[2]] for r in rows]); PY = np.array([[p[3] for p in r[2]] for r in rows])
    return X, Y, PA, PB, PX, PY

# =====================================================================
# SELFTEST
# =====================================================================
def selftest(args):
    print("=" * 78); print("CHERRY-PICK phase 1 SELF-TEST (no torch)"); print("=" * 78)
    rng = np.random.default_rng(2)
    sents, mark_idx = make_session(rng, 2, 4, 3, 3)
    print("sample:", " ".join(sents))
    A, B, X, ans = make_probes(rng, sents, mark_idx, 1)[0]
    fmt = lambda s: f"timeout {s[0]} retries {s[1]} auth {AUTHS[s[2]]} cache {CACHES[s[3]]}"
    print(f"  apply(from=v{A}, to=v{B}, start=[{fmt(X)}])")
    print(f"     edits in span: {span_ops(sents, mark_idx, A, B)}")
    print(f"     -> {fmt(ans)}")
    print(f"\nvocab {VOCAB} words | probes per session {args.n_probes}")
    print(f"\n{'config':<14}{'tokens':>10}{'edits A..B':>12} | {'labels':>8} | "
          + " | ".join(f"{b:^17}" for b in BASELINES) + " | order matters")
    print(f"{'':<14}{'':>10}{'(mean)':>12} | {'verified':>8} | " + " | ".join(f"{'exact  per-set':^17}" for _ in BASELINES) + " |")
    for i, (name, cfg) in enumerate(CONFIGS.items()):
        rows = build(400, 50 + i, cfg, args.n_probes)
        ok = tot = 0; spans = []
        for sents, mi, probes in rows:
            ok_stream = replay_states(sents) == gen_states(sents)          # two independent code paths
            for A, B, X, ans in probes:
                tot += 1; ok += cherry_verify(sents, A, B, X) == ans and ok_stream
                spans.append(len(span_ops(sents, mi, A, B)))
        assert ok == tot, f"{tot - ok} probe labels disagree with the independent re-parse"
        cells = []; om = []
        for bname, fn in BASELINES.items():
            ex = []; ps = []
            for sents, mi, probes in rows:
                for A, B, X, ans in probes:
                    pred = fn(sents, mi, A, B, X); ex.append(pred == ans); ps.append(np.mean([p == a for p, a in zip(pred, ans)]))
                    if bname == "order-blind": om.append(pred != ans)
            cells.append(f"{np.mean(ex):.2f}   {np.mean(ps):.2f}")
        X_, Y_, *_ = encode(rows[:50]); lens = (X_ > 0).sum(1)
        print(f"{name:<14}{f'{lens.min()}-{lens.max()}':>10}{np.mean(spans):>12.1f} | {'all' if ok == tot else 'FAIL':>8} | "
              + " | ".join(f"{c:^17}" for c in cells) + f" | {np.mean(om):.2f}")
    # dense labels: every turn end labelled once all four settings are set
    X_, Y_, *_ = encode(build(20, 9, CONFIGS["in-dist"], args.n_probes))
    print(f"\n[ok] dense stream labels at turn ends: {int((Y_[..., 0] >= 0).sum())} labelled turns in 20 sessions")
    print("  exact = all four settings right; order matters = share of probes where ignoring order changes the answer")

# =====================================================================
# TORCH (Colab)
# =====================================================================
def run(args):
    import torch, torch.nn as nn
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    MHA = importlib.import_module(args.attn_module).PathOrderedGeometricMHA
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

    class Memory(nn.Module):
        """the model being tested: embeddings + core + 4 stream heads"""
        def __init__(self, arm, std_layers=None, gru_hidden=None):
            super().__init__()
            self.kind = "holo" if arm.startswith("holo") else arm.split(":")[0]
            self.embed = nn.Embedding(VOCAB, d)
            self.width = gru_hidden or d
            self.heads = nn.ModuleList([nn.Linear(self.width, n) for n in N_CLS])
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

    class Probe(nn.Module):
        """apply(from=A, to=B, start=X): identical for every model, deliberately small"""
        def __init__(self, width):
            super().__init__()
            self.proj = nn.Identity() if width == d else nn.Linear(width, d)
            self.xemb = nn.ModuleList([nn.Embedding(n, d) for n in N_CLS])
            self.qmlp = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.attn = nn.MultiheadAttention(d, args.n_heads, batch_first=True)
            self.ln = nn.LayerNorm(d)
            self.heads = nn.ModuleList([nn.Linear(d, n) for n in N_CLS])
        def forward(self, mem, H, pad, PA, PB, PX):
            eA = mem.embed(torch.as_tensor([[W2I[f"v{a}"] for a in row] for row in PA.tolist()], device=H.device))
            eB = mem.embed(torch.as_tensor([[W2I[f"v{b}"] for b in row] for row in PB.tolist()], device=H.device))
            eX = sum(self.xemb[k](PX[..., k]) for k in range(4))
            q = self.qmlp(torch.cat([eA, eB, eX], -1))                       # (B, P, d)
            c, _ = self.attn(q, self.proj(H), self.proj(H), key_padding_mask=pad)
            o = self.ln(c + q)
            return [hd(o) for hd in self.heads]                               # 4 x (B, P, n)

    count = lambda m: sum(p.numel() for p in m.parameters())
    holo = [a for a in args.arms if a.startswith("holo")]
    target = count(Memory(holo[0])) if holo else None
    def make(arm):
        if arm.startswith("standard"):
            if ":" in arm:
                return Memory(arm, std_layers=int(arm.split(":")[1]))
            if target is None:
                raise SystemExit("standard needs a size: add a holo arm to match, or use standard:<layers>")
            n = min(range(1, 25), key=lambda n: abs(count(Memory("standard", std_layers=n)) - target))
            return Memory("standard", std_layers=n)
        if arm.startswith("gru"):
            if ":" in arm:
                return Memory("gru", gru_hidden=int(arm.split(":")[1]))
            if target is None:
                raise SystemExit("gru needs a size: add a holo arm to match, or use gru:<hidden>")
            w = min([32, 48, 64, 96, 128, 160, 192, 256, 320], key=lambda w: abs(count(Memory("gru", gru_hidden=w)) - target))
            return Memory("gru", gru_hidden=w)
        return Memory(arm)

    rows_tr = build(args.n_sess, args.seed + 100, CONFIGS["in-dist"], args.n_probes)
    nv = len(rows_tr) // 10
    val = encode(rows_tr[:nv]); tr_rows = rows_tr[nv:]
    tests = {name: (rows, encode(rows)) for i, (name, cfg) in enumerate(CONFIGS.items())
             for rows in [build(args.n_test_sess, 700 + i, cfg, args.n_probes)]}
    t = lambda z: torch.from_numpy(np.asarray(z)).to(DEVICE)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    @torch.no_grad()
    def evaluate(mem, probe, data, bs=32):
        X, Y, PA, PB, PX, PY = data; mem.eval(); probe.eval()
        ex, ps, st_ok, st_n = [], [], 0, 0
        for i in range(0, len(X), bs):
            xb = t(X[i:i + bs]); H = mem.hidden(xb); pad = xb == 0
            outs = probe(mem, H, pad, PA[i:i + bs], PB[i:i + bs], t(PX[i:i + bs]))
            pred = np.stack([o.argmax(-1).cpu().numpy() for o in outs], -1)  # (b, P, 4)
            gold = PY[i:i + bs]
            ex.extend((pred == gold).all(-1).ravel()); ps.extend((pred == gold).mean(-1).ravel())
            for k, hd in enumerate(mem.heads):
                yk = Y[i:i + bs, :, k]; m = yk >= 0
                pk = hd(H).argmax(-1).cpu().numpy()
                st_ok += (pk[m] == yk[m]).sum(); st_n += m.sum()
        return float(np.mean(ex)), float(np.mean(ps)), st_ok / st_n

    res = {}
    for arm in args.arms:
        mem = make(arm).to(DEVICE); probe = Probe(mem.width).to(DEVICE)
        print(f"\n--- {arm} --- memory params {count(mem):,}" + (f" ({100 * count(mem) / target:.0f}% of target)" if target else "")
              + f" | probe params {count(probe):,}")
        opt = torch.optim.Adam(list(mem.parameters()) + list(probe.parameters()), lr=args.lr)
        best, best_state = -1, None
        for ep in range(args.epochs):
            mem.train(); probe.train(); perm = np.random.permutation(len(tr_rows))
            for i in range(0, len(perm), args.bs):
                X, Y, PA, PB, PX, PY = encode([tr_rows[j] for j in perm[i:i + args.bs]])
                xb = t(X); opt.zero_grad()
                H = mem.hidden(xb); Yt = t(Y)
                l_stream = sum(ce(hd(H).reshape(-1, n), Yt[..., k].reshape(-1)) for k, (hd, n) in enumerate(zip(mem.heads, N_CLS)))
                outs = probe(mem, H, xb == 0, PA, PB, t(PX)); PYt = t(PY)
                l_probe = sum(ce(o.reshape(-1, n), PYt[..., k].reshape(-1)) for k, (o, n) in enumerate(zip(outs, N_CLS)))
                loss = l_stream + args.probe_weight * l_probe
                loss.backward(); torch.nn.utils.clip_grad_norm_(list(mem.parameters()) + list(probe.parameters()), 1.0); opt.step()
            vex, vps, vst = evaluate(mem, probe, val)
            if vps > best:
                best, best_state = vps, (copy.deepcopy(mem.state_dict()), copy.deepcopy(probe.state_dict()))
            if ep == 0 or (ep + 1) % 5 == 0:
                print(f"  ep {ep + 1}/{args.epochs} loss={loss.item():.3f} (stream {l_stream.item():.3f}, probe {l_probe.item():.3f}) "
                      f"val: cherry exact {vex:.3f}, per-setting {vps:.3f}, stream turns {vst:.3f}")
        mem.load_state_dict(best_state[0]); probe.load_state_dict(best_state[1])
        res[arm] = {name: evaluate(mem, probe, data) for name, (rows, data) in tests.items()}
        for name, (ex, ps, st) in res[arm].items():
            print(f"  [{arm}] {name:<13} cherry exact {ex:.3f} | per-setting {ps:.3f} | stream turns {st:.3f}")

    print("\n" + "=" * 78); print("CHERRY-PICK EXACT MATCH (all four settings right)")
    names = list(tests)
    print("arm".ljust(16) + "".join(n.rjust(15) for n in names))
    for bname, fn in BASELINES.items():
        row = []
        for n in names:
            rows = tests[n][0]
            row.append(np.mean([fn(s, mi, A, B, X) == ans for s, mi, pr in rows for A, B, X, ans in pr]))
        print(bname.ljust(16) + "".join(f"{v:.3f}".rjust(15) for v in row))
    for arm, r in res.items():
        print(arm.ljust(16) + "".join(f"{r[n][0]:.3f}".rjust(15) for n in names))
    print("\nCHERRY-PICK PER-SETTING ACCURACY (mean of the four)")
    for arm, r in res.items():
        print(arm.ljust(16) + "".join(f"{r[n][1]:.3f}".rjust(15) for n in names))
    print("\nSTREAM TURN-STATE ACCURACY (dense labels)")
    for arm, r in res.items():
        print(arm.ljust(16) + "".join(f"{r[n][2]:.3f}".rjust(15) for n in names))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--arms", nargs="+", default=["holo3:0.05", "standard", "gru"],
                   help="holo<L>:<edge_scale>, standard[:layers] (auto-matched), gru[:hidden] (auto-matched)")
    p.add_argument("--attn-module", dest="attn_module", default="attention_fast_chunked")
    p.add_argument("--n-sess", dest="n_sess", type=int, default=20000)
    p.add_argument("--n-test-sess", dest="n_test_sess", type=int, default=300)
    p.add_argument("--n-probes", dest="n_probes", type=int, default=8)
    p.add_argument("--probe-weight", dest="probe_weight", type=float, default=1.0)
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
