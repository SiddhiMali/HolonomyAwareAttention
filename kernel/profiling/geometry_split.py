"""
geometry_split.py -- split the geometry cost into its pieces so we know where
the time actually goes:  edge_mlp  vs  so_exp(matrix_exp)  vs  prefix-scan loop.

We already know:
  - geometry is ~89% of holo's total cost (earlier profile)
  - matrix_exp is NOT beatable at d=16, edge_scale=1.5 (exp_compare.py)
So the question now: is so_exp the dominant piece, or is the prefix-scan loop?
If the loop dominates -> optimizing the scan is worthwhile.
If so_exp dominates -> the geometry is basically irreducible (matrix_exp is optimal),
   and the answer is "run fewer/smaller steps", not "optimize".

Run:  python geometry_split.py
"""
import os as _os, sys as _sys
_K = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # kernel/
_ROOT = _os.path.dirname(_K)                                           # repo root
for _p in (_K, _ROOT):
    if _p not in _sys.path:
        _sys.path.append(_p)
import time
import torch

from attention_fast_chunked import PathOrderedGeometricMHA

DEVICE = "cuda"


def sync():
    torch.cuda.synchronize()


def time_pieces(B, L, D, Hh, edge_scale, iters=50, warmup=10):
    m = PathOrderedGeometricMHA(D, n_heads=Hh, mode="causal",
                                edge_scale=edge_scale).to(DEVICE).eval()
    x = torch.randn(B, L, D, device=DEVICE)
    H_, d = m.h, m.d
    m_dim = m.m_dim

    # reproduce the pieces of _geometry separately, timing each.
    from utils import so_exp, skew_from_vec

    def piece_edge_mlp():
        x_right = torch.roll(x, shifts=-1, dims=1)
        pair = torch.cat([x, x_right], dim=-1)
        a_all = m.edge_mlp(pair).view(B, L, H_, m_dim).transpose(1, 2) * m.edge_scale
        return a_all

    a_all = piece_edge_mlp()
    a_adj = a_all[:, :, :-1]

    def piece_so_exp():
        return so_exp(skew_from_vec(a_adj, d))

    G_adj = piece_so_exp()

    def piece_prefix_scan():
        I = torch.eye(d, device=DEVICE)
        P = torch.empty(B, H_, L, d, d, device=DEVICE)
        P[:, :, 0] = I
        Pt = I.expand(B, H_, d, d).clone()
        for t in range(L - 1):
            Pt = G_adj[:, :, t] @ Pt
            P[:, :, t + 1] = Pt
        return P

    def bench(fn):
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        return (time.perf_counter() - t0) / iters * 1e3

    t_mlp = bench(piece_edge_mlp)
    t_exp = bench(piece_so_exp)
    t_scan = bench(piece_prefix_scan)
    total = t_mlp + t_exp + t_scan
    return t_mlp, t_exp, t_scan, total


if __name__ == "__main__":
    edge_scale = 1.5
    B, D, Hh = 8, 64, 4
    print(f"edge_scale={edge_scale}, B={B}, D={D}, heads={Hh}\n")
    print(f"{'L':>5}{'edge_mlp':>10}{'so_exp':>10}{'prefix':>10}{'total':>10}"
          f"{'  mlp%':>7}{'exp%':>7}{'scan%':>7}")
    for L in (16, 32, 64, 128):
        mlp, exp, scan, tot = time_pieces(B, L, D, Hh, edge_scale)
        print(f"{L:>5}{mlp:>10.3f}{exp:>10.3f}{scan:>10.3f}{tot:>10.3f}"
              f"{100*mlp/tot:>6.0f}%{100*exp/tot:>6.0f}%{100*scan/tot:>6.0f}%")
    print("\nREAD:")
    print("  so_exp% high  -> geometry near-irreducible (matrix_exp is optimal);")
    print("                   answer is fewer/smaller steps, not optimization.")
    print("  scan% high    -> the sequential prefix loop is the target;")
    print("                   worth de-Pythonizing / parallel-scan.")
    print("  mlp% high     -> unusual; the MLP itself dominates.")
