"""
time_holo.py -- find out WHERE holo's time actually goes.

Splits one holo forward into:
  (1) geometry: prefix scan (Python loop) + matrix_exp  -> the _geometry() call
  (2) attention: the tiled fold loop                    -> the rest of forward()

Run in Colab:  !python time_holo.py
or paste into a cell.

Tells you whether a Triton ATTENTION kernel would even help holo, or whether
the bottleneck is the geometry (prefix scan / matrix_exp), which a kernel does
NOT touch.
"""
import os as _os, sys as _sys
_K = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # kernel/
_ROOT = _os.path.dirname(_K)                                           # repo root
for _p in (_K, _ROOT):
    if _p not in _sys.path:
        _sys.path.append(_p)
import time
import torch

# use the SAME attention file your experiment uses
from attention_fast_chunked import PathOrderedGeometricMHA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_forward(L, B=128, D=64, H=4, edge_scale=1.5, iters=50, warmup=10):
    """Time a full holo forward and, separately, just the geometry setup."""
    m = PathOrderedGeometricMHA(D, n_heads=H, mode="causal",
                                edge_scale=edge_scale).to(DEVICE).eval()
    x = torch.randn(B, L, D, device=DEVICE)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=DEVICE))

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            m(x, attn_mask=mask, return_geometry=False)
            m._geometry(x)
    sync()

    # time the geometry setup alone (prefix scan + matrix_exp)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            m._geometry(x)
    sync()
    t_geom = (time.perf_counter() - t0) / iters * 1e3  # ms

    # time the FULL forward (geometry + attention)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            m(x, attn_mask=mask, return_geometry=False)
    sync()
    t_full = (time.perf_counter() - t0) / iters * 1e3  # ms

    t_attn = t_full - t_geom  # attention loop = full minus geometry
    return t_full, t_geom, t_attn


if __name__ == "__main__":
    print(f"device = {DEVICE}")
    print(f"{'L':>5} {'full ms':>9} {'geom ms':>9} {'attn ms':>9} "
          f"{'geom %':>7} {'attn %':>7}")
    for L in (7, 14, 21, 32, 64):
        full, geom, attn = time_forward(L)
        gp = 100 * geom / full if full > 0 else 0
        ap = 100 * attn / full if full > 0 else 0
        print(f"{L:>5} {full:>9.3f} {geom:>9.3f} {attn:>9.3f} "
              f"{gp:>6.1f}% {ap:>6.1f}%")
    print("\nREAD: if geom% is large at your L (~21), a Triton ATTENTION kernel")
    print("won't help much -- the cost is the prefix scan + matrix_exp, which")
    print("live OUTSIDE the attention grid. If attn% dominates, a kernel would help.")
