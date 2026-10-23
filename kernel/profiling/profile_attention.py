"""
Equivalence + throughput harness for PathOrderedGeometricMHA.

Usage:
    # put your real utils.py on the path (it must define so_exp, skew_from_vec)
    python profile_attention.py

If you don't have utils.py handy, this file installs a reference version into
sys.modules so both the original and fast modules import the *same* geometry
ops -- equivalence is then a fair comparison. Swap REFERENCE_UTILS = False to
use your own utils.py instead.
"""
import os as _os, sys as _sys
_K = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # kernel/
_ROOT = _os.path.dirname(_K)                                           # repo root
for _p in (_K, _ROOT):
    if _p not in _sys.path:
        _sys.path.append(_p)
import sys, types, time, math, importlib.util
import torch

REFERENCE_UTILS = False  # use the repo's utils.py (set True for a standalone copy)

if REFERENCE_UTILS:
    u = types.ModuleType("utils")

    def skew_from_vec(a, d):
        # a: (..., m) with m = d(d-1)/2 -> (..., d, d) skew-symmetric
        idx = torch.triu_indices(d, d, offset=1, device=a.device)
        S = torch.zeros(*a.shape[:-1], d, d, device=a.device, dtype=a.dtype)
        S[..., idx[0], idx[1]] = a
        S = S - S.transpose(-1, -2)
        return S

    def so_exp(S):
        return torch.matrix_exp(S)

    u.skew_from_vec = skew_from_vec
    u.so_exp = so_exp
    sys.modules["utils"] = u


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# adjust paths as needed
ref = load(_os.path.join(_ROOT, "attention.py"), "attention_ref")
fast = load(_os.path.join(_K, "attention_fast.py"), "attention_fast")


def copy_weights(src, dst):
    dst.load_state_dict(src.state_dict())


def equivalence(B, L, D, heads, mode, tol=1e-4, use_ring_mask=False):
    torch.manual_seed(0)
    x = torch.randn(B, L, D)
    mask = None
    if use_ring_mask:
        try:
            from utils import build_ring_attention_mask
            mask = build_ring_attention_mask(L, x.device)
        except Exception:
            pass
    m_ref = ref.PathOrderedGeometricMHA(D, heads, dropout=0.0, mode=mode).eval()
    m_fast = fast.PathOrderedGeometricMHA(D, heads, dropout=0.0, mode=mode).eval()
    copy_weights(m_ref, m_fast)
    with torch.no_grad():
        y_ref, g_ref = m_ref(x, attn_mask=mask, return_geometry=True)
        y_fast, g_fast = m_fast(x, attn_mask=mask, return_geometry=True)
    dy = (y_ref - y_fast).abs().max().item()
    da = (g_ref["attn"] - g_fast["attn"]).abs().max().item()
    dh = (g_ref["H"] - g_fast["H"]).abs().max().item()
    ok = max(dy, da, dh) < tol
    print(f"[{mode:6}] L={L:3} max|dy|={dy:.2e} max|dattn|={da:.2e} "
          f"max|dH|={dh:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def bench(model, x, iters=20, warmup=5):
    cuda = x.is_cuda
    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def throughput(B, D, heads, mode, lengths):
    print(f"\n=== throughput  mode={mode}  B={B} D={D} H={heads} ===")
    print(f"{'L':>5} {'ref ms':>10} {'fast ms':>10} {'speedup':>9}")
    for L in lengths:
        x = torch.randn(B, L, D)
        m_ref = ref.PathOrderedGeometricMHA(D, heads, mode=mode).eval()
        m_fast = fast.PathOrderedGeometricMHA(D, heads, mode=mode).eval()
        copy_weights(m_ref, m_fast)
        with torch.no_grad():
            tr = bench(m_ref, x)
            tf = bench(m_fast, x)
        print(f"{L:>5} {tr:>10.2f} {tf:>10.2f} {tr/tf:>8.1f}x")


if __name__ == "__main__":
    torch.set_num_threads(torch.get_num_threads())
    print("equivalence checks (dense mask):")
    for mode in ("causal", "inert"):
        for L in (4, 8, 16, 32):
            equivalence(B=2, L=L, D=32, heads=4, mode=mode)
    print("\nequivalence checks (ring mask):")
    for mode in ("causal", "inert"):
        for L in (8, 16, 32):
            equivalence(B=2, L=L, D=32, heads=4, mode=mode, use_ring_mask=True)

    throughput(B=4, D=64, heads=8, mode="causal", lengths=(16, 32, 64, 128))
    throughput(B=4, D=64, heads=8, mode="inert",  lengths=(16, 32, 64, 128))
