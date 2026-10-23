"""
skew_test.py -- is the "so_exp bottleneck" actually skew_from_vec's Python loop?

Hypothesis: utils.skew_from_vec does 240 tiny indexed assignments at d=16
(a double Python loop), each a separate GPU kernel launch. That fixed overhead,
not matrix_exp, may be most of the geometry cost.

This file:
  1. checks a vectorized skew_from_vec_fast gives the IDENTICAL matrix
  2. checks gradients are identical
  3. times: old skew, fast skew, matrix_exp alone, and so_exp(skew) both ways

Run:  python skew_test.py   (utils.py must be importable)
"""
import os as _os, sys as _sys
_K = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # kernel/
_ROOT = _os.path.dirname(_K)                                           # repo root
for _p in (_K, _ROOT):
    if _p not in _sys.path:
        _sys.path.append(_p)
import time
import torch
from utils import skew_from_vec, so_exp

DEVICE = "cuda"
d = 16
m_dim = d * (d - 1) // 2
_IU_CACHE = {}


def skew_from_vec_fast(v, d):
    """Vectorized drop-in: same upper-triangle ordering as the loop version
    (row-major: i outer, j > i inner), filled in ONE indexed op."""
    *prefix, m = v.shape
    if m != d * (d - 1) // 2:
        raise ValueError(f"Expected last dim {d*(d-1)//2}, got {m}")
    key = (d, v.device)
    if key not in _IU_CACHE:
        _IU_CACHE[key] = torch.triu_indices(d, d, offset=1, device=v.device)
    iu = _IU_CACHE[key]
    A = v.new_zeros(*prefix, d, d)
    A[..., iu[0], iu[1]] = v
    return A - A.transpose(-1, -2)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- 1. exactness ----
    v = torch.randn(8, 4, 63, m_dim, device=DEVICE) * 1.5
    A_old = skew_from_vec(v, d)
    A_new = skew_from_vec_fast(v, d)
    print(f"1. matrices identical:   max|diff| = {(A_old - A_new).abs().max().item():.1e}")

    # ---- 2. gradients ----
    va = v.clone().requires_grad_(True)
    vb = v.clone().requires_grad_(True)
    g = torch.randn_like(A_old)
    (skew_from_vec(va, d) * g).sum().backward()
    (skew_from_vec_fast(vb, d) * g).sum().backward()
    print(f"2. gradients identical:  max|diff| = {(va.grad - vb.grad).abs().max().item():.1e}")

    # ---- 3. timing at geometry_split sizes (B=8, H=4) ----
    print(f"\n3. timing (ms), B=8, H=4, d={d}")
    print(f"{'L':>5}{'old_skew':>10}{'fast_skew':>11}{'mexp_only':>11}"
          f"{'so_exp_old':>12}{'so_exp_new':>12}{'speedup':>9}")
    for L in (16, 32, 64, 128):
        vv = torch.randn(8, 4, L - 1, m_dim, device=DEVICE) * 1.5
        A = skew_from_vec_fast(vv, d)
        t_old = bench(lambda: skew_from_vec(vv, d))
        t_new = bench(lambda: skew_from_vec_fast(vv, d))
        t_mexp = bench(lambda: torch.matrix_exp(A))
        t_full_old = bench(lambda: so_exp(skew_from_vec(vv, d)))
        t_full_new = bench(lambda: so_exp(skew_from_vec_fast(vv, d)))
        print(f"{L:>5}{t_old:>10.3f}{t_new:>11.3f}{t_mexp:>11.3f}"
              f"{t_full_old:>12.3f}{t_full_new:>12.3f}{t_full_old/t_full_new:>8.2f}x")

    print("\nREAD: if old_skew >> mexp_only, the loop was the bottleneck, not")
    print("matrix_exp. If 1 and 2 are ~0, skew_from_vec_fast is a safe drop-in")
    print("replacement in utils.py (same output, same gradients).")
