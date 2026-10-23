"""
exp_compare.py -- test every matrix-exponential approach against the trusted
reference (torch.matrix_exp), on YOUR actual matrices (skew-symmetric, your d,
your edge_scale). Reports accuracy AND speed. Let the numbers pick the winner.

Approaches tested:
  0. reference        : torch.matrix_exp                  (ground truth)
  1. taylor_k         : truncated Taylor series, K terms  (a few matmuls)
  2. scale_square     : scaling + Taylor + squaring       (accurate for larger norms)
  3. eigh_based       : eigendecomposition route          (uses skew structure)

Run:  python exp_compare.py
Reads: nothing external except utils.skew_from_vec for realistic inputs.
"""
import os as _os, sys as _sys
_K = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # kernel/
_ROOT = _os.path.dirname(_K)                                           # repo root
for _p in (_K, _ROOT):
    if _p not in _sys.path:
        _sys.path.append(_p)
import math, time
import torch

# ---- config: MATCH YOUR EXPERIMENT ----
d = 16              # <-- set to your per-head d (d_model/n_heads). CHANGE ME.
edge_scale = 1.5    # <-- your edge_scale (0.1 or 1.5 etc). CHANGE ME.
B, H, L = 4, 4, 64  # batch*head*seq worth of matrices to exp at once
DEVICE = "cuda"
DTYPE = torch.float32

m_dim = d * (d - 1) // 2


def make_skew(n):
    """n random skew-symmetric d x d matrices, scaled like your real ones."""
    v = torch.randn(n, m_dim, device=DEVICE, dtype=DTYPE) * edge_scale
    A = torch.zeros(n, d, d, device=DEVICE, dtype=DTYPE)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[:, i, j] = v[:, idx]
            A[:, j, i] = -v[:, idx]
            idx += 1
    return A


# ============ APPROACHES ============
def ref_exp(A):
    return torch.matrix_exp(A)


def taylor_exp(A, K):
    """exp(A) = I + A + A^2/2! + ... + A^K/K!  (naive truncation)."""
    n = A.shape[0]
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand(n, d, d)
    out = I.clone()
    term = I.clone()
    for k in range(1, K + 1):
        term = torch.bmm(term, A) / k     # term = A^k / k!
        out = out + term
    return out


def scale_square_exp(A, K, s):
    """Scale A by 2^-s, Taylor-truncate, then square s times."""
    As = A / (2 ** s)
    E = taylor_exp(As, K)
    for _ in range(s):
        E = torch.bmm(E, E)
    return E


def eigh_exp(A):
    """
    Skew-symmetric: iA is Hermitian, so eigh applies to (iA).
    exp(A) = V @ diag(exp(eigvals)) @ V^H, taken real.
    """
    iA = (1j * A).to(torch.complex64)
    w, V = torch.linalg.eigh(iA)                 # w real, V unitary (eig of iA)
    # eigenvalues of A are -i*w  => exp(A) eigenvalues = exp(-i*w)
    expw = torch.exp(-1j * w)                     # (n, d) complex
    E = V @ torch.diag_embed(expw) @ V.conj().transpose(-1, -2)
    return E.real.to(A.dtype)


# ============ TIMING ============
def bench(fn, A, iters=50, warmup=10):
    for _ in range(warmup):
        fn(A)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(A)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def maxerr(E, Eref):
    return (E - Eref).abs().max().item()


if __name__ == "__main__":
    n = B * H * L
    A = make_skew(n)
    print(f"d={d}, edge_scale={edge_scale}, n_matrices={n}, dtype={DTYPE}")
    print(f"typical ||A|| (max abs entry): {A.abs().max().item():.3f}\n")

    Eref = ref_exp(A)
    t_ref = bench(ref_exp, A)
    print(f"{'approach':<22}{'max_err':>12}{'ms':>10}{'speedup':>10}")
    print(f"{'reference (matrix_exp)':<22}{0.0:>12.2e}{t_ref:>10.3f}{'1.00x':>10}")

    # taylor with several term counts
    for K in (4, 6, 8, 10, 12):
        E = taylor_exp(A, K)
        err = maxerr(E, Eref)
        t = bench(lambda a: taylor_exp(a, K), A)
        print(f"{'taylor K='+str(K):<22}{err:>12.2e}{t:>10.3f}{t_ref/t:>9.2f}x")

    # scale-square variants
    for (K, s) in ((6, 1), (6, 2), (8, 2), (4, 3)):
        E = scale_square_exp(A, K, s)
        err = maxerr(E, Eref)
        t = bench(lambda a: scale_square_exp(a, K, s), A)
        print(f"{'scalesq K='+str(K)+' s='+str(s):<22}{err:>12.2e}{t:>10.3f}{t_ref/t:>9.2f}x")

    # eigh
    E = eigh_exp(A)
    err = maxerr(E, Eref)
    t = bench(eigh_exp, A)
    print(f"{'eigh':<22}{err:>12.2e}{t:>10.3f}{t_ref/t:>9.2f}x")

    print("\nREAD: pick the FASTEST approach whose max_err is small enough")
    print("(e.g. < 1e-5). Accuracy AND speed both matter. If nothing beats")
    print("reference on speed, matrix_exp is already good -- look at the prefix")
    print("scan instead.")
