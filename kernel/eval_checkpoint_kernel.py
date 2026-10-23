"""
eval_checkpoint_kernel.py -- evaluation suite for the holonomy Triton KERNEL,
in the context of the checkpoint task.

Answers four questions, in order (each gates the next):

  1. CORRECTNESS  : does the kernel produce the SAME output as the verified
                    reference (attention_fast_chunked)?  (forward + gradients)
                    -> if this fails, nothing else matters.

  2. SPEED        : is the kernel FASTER, and how does the advantage scale with
                    sequence length L?  (proper warmup + cuda-event timing,
                    forward-only AND forward+backward, swept over L)

  3. MEMORY       : does the kernel use less PEAK GPU memory, swept over L?

  4. TASK         : does the checkpoint experiment give the SAME accuracy with
                    the kernel as with the reference?  (correctness in a real run)

REQUIRES (GPU runtime):
  kernel/: attention_kernel.py (self-contained kernel), attention_fast_chunked.py,
           attention_fast.py
  repo root: utils.py, attention.py
  + for the TASK section: qgate_comparev5.py, qgate_checkpoint.py, qgate_data.py,
    qgate_energy.py  (and whatever they import)

USAGE:
  # everything:
  python eval_checkpoint_kernel.py --all
  # just the fast checks (no full training):
  python eval_checkpoint_kernel.py --correctness --speed --memory
  # only the task run (slow):
  python eval_checkpoint_kernel.py --task --epochs 30

NOTE: run the module self-test FIRST (verifies the kernel in isolation):
  python kernel/attention_kernel.py    # expect: ALL OK
This suite assumes those already pass; it evaluates the *improvement*.
"""
from __future__ import annotations
import argparse, time, gc
import torch
import os as _os, sys as _sys
# repo root (parent of kernel/) holds utils.py and attention.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# the two implementations under comparison (identical interface, identical weights)
from attention_fast_chunked import PathOrderedGeometricMHA as RefMHA
from attention_kernel import PathOrderedGeometricMHA as KerMHA


# =====================================================================
# helpers
# =====================================================================
def _make_pair(d_model, n_heads, edge_scale, seed=0):
    """Two modules (ref, kernel) with IDENTICAL weights -> fair comparison."""
    torch.manual_seed(seed)
    ref = RefMHA(d_model, n_heads=n_heads, mode="causal", edge_scale=edge_scale).to(DEVICE)
    ker = KerMHA(d_model, n_heads=n_heads, mode="causal", edge_scale=edge_scale).to(DEVICE)
    ker.load_state_dict(ref.state_dict())     # same weights: only the impl differs
    return ref, ker


def _inputs(B, L, d_model, requires_grad=False):
    x = torch.randn(B, L, d_model, device=DEVICE, requires_grad=requires_grad)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=DEVICE))
    return x, mask


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# =====================================================================
# 1. CORRECTNESS  (kernel output == reference output)
# =====================================================================
def check_correctness(d_model=64, n_heads=4, edge_scale=1.5,
                      lengths=(8, 16, 32, 64), B=4, atol=1e-3):
    print("\n" + "=" * 64)
    print("1. CORRECTNESS  (kernel vs reference: same output + same grads)")
    print("=" * 64)
    all_ok = True
    for L in lengths:
        ref, ker = _make_pair(d_model, n_heads, edge_scale)
        ref.eval(); ker.eval()

        # --- forward match ---
        x, mask = _inputs(B, L, d_model)
        with torch.no_grad():
            y_ref, _ = ref(x, attn_mask=mask, return_geometry=False)
            y_ker, _ = ker(x, attn_mask=mask, return_geometry=False)
        dy = (y_ref - y_ker).abs().max().item()

        # --- gradient match (same input, backprop a scalar) ---
        x1 = x.clone().detach().requires_grad_(True)
        x2 = x.clone().detach().requires_grad_(True)
        yr, _ = ref(x1, attn_mask=mask, return_geometry=False); yr.sum().backward()
        yk, _ = ker(x2, attn_mask=mask, return_geometry=False); yk.sum().backward()
        dg = (x1.grad - x2.grad).abs().max().item()

        ok = (dy < atol) and (dg < atol)
        all_ok &= ok
        print(f"  L={L:4d}  max|dy|={dy:.2e}  max|d(grad)|={dg:.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    print(f"\n  CORRECTNESS: {'ALL OK -- kernel matches reference' if all_ok else 'FAIL -- do NOT trust speed/task numbers'}")
    return all_ok


# =====================================================================
# 2. SPEED  (latency, swept over L, warmup + cuda-events, fwd and fwd+bwd)
# =====================================================================
def _time_forward(model, x, mask, n_warmup, n_iter):
    model.eval()
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x, attn_mask=mask, return_geometry=False)
    _sync()
    if DEVICE == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iter):
            with torch.no_grad():
                model(x, attn_mask=mask, return_geometry=False)
        e.record(); _sync()
        return s.elapsed_time(e) / n_iter          # ms
    else:
        t0 = time.time()
        for _ in range(n_iter):
            with torch.no_grad():
                model(x, attn_mask=mask, return_geometry=False)
        return (time.time() - t0) / n_iter * 1000


def _time_fwd_bwd(model, x, mask, n_warmup, n_iter):
    model.train()
    for _ in range(n_warmup):
        y, _ = model(x, attn_mask=mask, return_geometry=False)
        y.sum().backward(); model.zero_grad(); x.grad = None
    _sync()
    if DEVICE == "cuda":
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iter):
            y, _ = model(x, attn_mask=mask, return_geometry=False)
            y.sum().backward(); model.zero_grad(); x.grad = None
        e.record(); _sync()
        return s.elapsed_time(e) / n_iter
    else:
        t0 = time.time()
        for _ in range(n_iter):
            y, _ = model(x, attn_mask=mask, return_geometry=False)
            y.sum().backward(); model.zero_grad(); x.grad = None
        return (time.time() - t0) / n_iter * 1000


def bench_speed(d_model=64, n_heads=4, edge_scale=1.5, B=8,
                lengths=(32, 64, 128, 256, 512), n_warmup=10, n_iter=50):
    print("\n" + "=" * 64)
    print("2. SPEED  (ms/forward; lower=better; watch the trend with L)")
    print("=" * 64)
    print(f"  batch={B}, d_model={d_model}, heads={n_heads}\n")
    print(f"  {'L':>5} {'ref_fwd':>10} {'ker_fwd':>10} {'fwd_speedup':>12}"
          f" {'ref_f+b':>10} {'ker_f+b':>10} {'f+b_speedup':>12}")
    for L in lengths:
        ref, ker = _make_pair(d_model, n_heads, edge_scale)
        try:
            x, mask = _inputs(B, L, d_model)
            rf = _time_forward(ref, x, mask, n_warmup, n_iter)
            kf = _time_forward(ker, x, mask, n_warmup, n_iter)

            xg1, _ = _inputs(B, L, d_model, requires_grad=True)
            xg2 = xg1.clone().detach().requires_grad_(True)
            rb = _time_fwd_bwd(ref, xg1, mask, max(3, n_warmup // 2), max(10, n_iter // 2))
            kb = _time_fwd_bwd(ker, xg2, mask, max(3, n_warmup // 2), max(10, n_iter // 2))

            print(f"  {L:>5} {rf:>10.3f} {kf:>10.3f} {rf/kf:>11.2f}x"
                  f" {rb:>10.3f} {kb:>10.3f} {rb/kb:>11.2f}x")
        except RuntimeError as err:  # e.g. OOM at large L
            print(f"  {L:>5}  RuntimeError (likely OOM): {str(err)[:50]}")
        del ref, ker; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    print("\n  READ: forward speedup is the kernel's main claim; it typically GROWS")
    print("  with L. fwd+bwd includes the recompute-backward cost (kernel fwd is")
    print("  fast, but backward recomputes in PyTorch), so f+b speedup is smaller")
    print("  or <1 -- that's honest. Report both.")


# =====================================================================
# 3. MEMORY  (peak GPU memory, swept over L)
# =====================================================================
def _peak_mem_forward(model, x, mask):
    if DEVICE != "cuda":
        return float("nan")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    with torch.no_grad():
        model(x, attn_mask=mask, return_geometry=False)
    _sync()
    return torch.cuda.max_memory_allocated() / 1e6   # MB


def bench_memory(d_model=64, n_heads=4, edge_scale=1.5, B=8,
                 lengths=(32, 64, 128, 256, 512)):
    print("\n" + "=" * 64)
    print("3. MEMORY  (peak MB during forward; lower=better)")
    print("=" * 64)
    if DEVICE != "cuda":
        print("  (no CUDA -> skipping memory benchmark)")
        return
    print(f"  {'L':>5} {'ref_MB':>10} {'ker_MB':>10} {'ratio(ref/ker)':>16}")
    for L in lengths:
        ref, ker = _make_pair(d_model, n_heads, edge_scale); ref.eval(); ker.eval()
        try:
            x, mask = _inputs(B, L, d_model)
            rm = _peak_mem_forward(ref, x, mask)
            km = _peak_mem_forward(ker, x, mask)
            print(f"  {L:>5} {rm:>10.1f} {km:>10.1f} {rm/max(km,1e-6):>15.2f}x")
        except RuntimeError as err:
            print(f"  {L:>5}  RuntimeError (likely OOM): {str(err)[:50]}")
        del ref, ker; gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# 4. TASK  (checkpoint experiment; does the kernel give the same accuracy?)
# =====================================================================
def run_task(args):
    """Run the checkpoint task with the KERNEL holo (and optionally reference holo)
    and print accuracy, to confirm the kernel is correct in a REAL training run.

    We import the checkpoint harness pieces and monkey-patch which
    PathOrderedGeometricMHA the Model uses, so we can run kernel vs reference
    holo under identical everything else."""
    print("\n" + "=" * 64)
    print("4. TASK  (checkpoint experiment: kernel-holo accuracy)")
    print("=" * 64)
    import numpy as np
    # import harness pieces
    import qgate_comparev5 as qc
    from qgate_checkpoint import (make_gate_bank, h2_like_hamiltonian,
                                  make_dataset_checkpoint, calibrate_bins_checkpoint,
                                  vocab_size)
    # the checkpoint training code (reuse from the uploaded file if importable),
    # else inline a minimal train/eval:
    import torch.nn as nn

    class QueryReadoutModel(nn.Module):
        def __init__(self, kind, vocab, n_bins, **kw):
            super().__init__(); self.inner = qc.Model(kind, vocab, n_bins, **kw)
        def forward(self, x):
            h = self.inner.encode(x); last = h[:, -1]
            return self.inner.cls_head(last), self.inner.reg_head(last).squeeze(-1)

    def evaluate(model, X, Y, E, n_bins, bs=128):
        model.eval(); corr = tot = 0
        with torch.no_grad():
            for i in range(0, X.shape[0], bs):
                xb, yb = X[i:i+bs].to(DEVICE), Y[i:i+bs].to(DEVICE)
                logits, _ = model(xb)
                corr += (logits.argmax(-1) == yb).sum().item(); tot += yb.numel()
        return corr / max(tot, 1)

    def train_holo(vocab, n_bins, tr, tests, edges, which_impl):
        # monkey-patch the impl that qgate_comparev5.Model uses for holo
        import attention_fast_chunked, attention_kernel
        impl = (attention_kernel.PathOrderedGeometricMHA if which_impl == "kernel"
                else attention_fast_chunked.PathOrderedGeometricMHA)
        # qc.Model builds holo core from a symbol it imported; patch that symbol.
        # (Assumes qgate_comparev5 imported PathOrderedGeometricMHA at module level.)
        qc.PathOrderedGeometricMHA = impl
        Xtr, Ytr, Etr = tr
        model = QueryReadoutModel("holo", vocab, n_bins, d_model=args.d_model,
                                  n_heads=args.n_heads, n_std_layers=args.n_std_layers,
                                  edge_scale=args.edge_scale).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
        n = Xtr.shape[0]; nval = max(1, n // 10)
        Xv, Yv, Ev = Xtr[:nval], Ytr[:nval], Etr[:nval]
        Xt, Yt, Et = Xtr[nval:], Ytr[nval:], Etr[nval:]
        nt = Xt.shape[0]; best = -1.0; import copy; best_state = None
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(nt)
            for i in range(0, nt, args.bs):
                idx = perm[i:i+args.bs]
                xb, yb, eb = Xt[idx].to(DEVICE), Yt[idx].to(DEVICE), Et[idx].to(DEVICE)
                opt.zero_grad(); logits, epred = model(xb)
                loss = ce(logits, yb) + mse(epred, eb); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            acc = evaluate(model, Xv, Yv, Ev, n_bins)
            if acc > best: best = acc; best_state = copy.deepcopy(model.state_dict())
            if (ep+1) % 5 == 0 or ep == 0:
                print(f"    [{which_impl}] ep {ep+1}/{args.epochs} val_acc={acc:.3f} best={best:.3f}")
        if best_state: model.load_state_dict(best_state)
        out = {}
        for L, ts in tests.items():
            out[L] = evaluate(model, *ts, n_bins)
        return out

    # build data
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gates_c = make_gate_bank(args.n_qubits, args.K, seed=args.gate_seed)
    H = h2_like_hamiltonian(); vocab = vocab_size(args.K)
    edges = calibrate_bins_checkpoint(args.train_len, args.K, args.p_ckpt, gates_c, H, args.n_bins)
    def to_t(X, Y, E): return (torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(E))
    tr = to_t(*make_dataset_checkpoint(args.n_train, args.train_len, args.K,
                                       args.p_ckpt, gates_c, H, edges, seed=args.seed))
    tests = {L: to_t(*make_dataset_checkpoint(args.n_test, L, args.K, args.p_ckpt,
                     gates_c, H, edges, seed=1000+L)) for L in args.test_lens}

    print(f"\n  running holo with KERNEL impl ...")
    ker_out = train_holo(vocab, args.n_bins, tr, tests, edges, "kernel")
    print(f"\n  === kernel-holo accuracy by length ===")
    for L in args.test_lens:
        tag = "train" if L == args.train_len else "extrap"
        print(f"    len={L:3d} ({tag})  acc={ker_out[L]:.3f}")

    if args.task_compare:
        print(f"\n  running holo with REFERENCE impl (for parity check) ...")
        ref_out = train_holo(vocab, args.n_bins, tr, tests, edges, "reference")
        print(f"\n  === kernel vs reference holo (should be CLOSE) ===")
        print(f"    {'len':>5} {'kernel':>9} {'reference':>10} {'diff':>8}")
        for L in args.test_lens:
            print(f"    {L:>5} {ker_out[L]:>9.3f} {ref_out[L]:>10.3f} {abs(ker_out[L]-ref_out[L]):>8.3f}")
        print("\n  (small diffs = training noise; large diffs = kernel changes the")
        print("   result -> investigate. Same numbers = kernel correct in a real run.)")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--task", action="store_true")
    ap.add_argument("--task-compare", action="store_true",
                    help="also train reference-holo to compare accuracy parity")
    # model / bench params
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-std-layers", type=int, default=4)
    ap.add_argument("--edge-scale", type=float, default=1.5)
    ap.add_argument("--bench-lens", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--bench-batch", type=int, default=8)
    # task params
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-qubits", type=int, default=2)
    ap.add_argument("--p-ckpt", type=float, default=0.2)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--train-len", type=int, default=16)
    ap.add_argument("--test-lens", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-seed", type=int, default=0)
    args = ap.parse_args()

    if DEVICE != "cuda":
        print("!!! WARNING: no CUDA. The kernel REQUIRES a GPU. Correctness/task may")
        print("!!! run via the reference fallback; speed/memory are meaningless on CPU.")

    do = lambda flag: args.all or getattr(args, flag)

    ok = True
    if do("correctness"):
        ok = check_correctness(args.d_model, args.n_heads, args.edge_scale)
        if not ok:
            print("\n  Correctness FAILED -> stopping (speed/task on a wrong kernel are meaningless).")
            return
    if do("speed"):
        bench_speed(args.d_model, args.n_heads, args.edge_scale,
                    B=args.bench_batch, lengths=tuple(args.bench_lens))
    if do("memory"):
        bench_memory(args.d_model, args.n_heads, args.edge_scale,
                     B=args.bench_batch, lengths=tuple(args.bench_lens))
    if do("task"):
        run_task(args)

    print("\nDONE.")


if __name__ == "__main__":
    main()
