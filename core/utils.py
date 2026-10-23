import torch


def skew_from_vec(v: torch.Tensor, d: int) -> torch.Tensor:
    """
    Convert vectorized upper-triangular entries into a skew-symmetric matrix.

    Args:
        v: (..., m_dim) where m_dim = d * (d - 1) // 2
        d: matrix size

    Returns:
        A: (..., d, d) skew-symmetric
    """
    *prefix, m_dim = v.shape
    expected = d * (d - 1) // 2
    if m_dim != expected:
        raise ValueError(f"Expected last dim {expected}, got {m_dim}")

    A = v.new_zeros(*prefix, d, d)
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[..., i, j] = v[..., idx]
            A[..., j, i] = -v[..., idx]
            idx += 1
    return A


def so_exp(A: torch.Tensor) -> torch.Tensor:
    """
    Skew-symmetric -> SO(d) via the MATRIX EXPONENTIAL, expm(A).

    WHY expm AND NOT THE CAYLEY TRANSFORM:
      This map was previously the Cayley transform Q = (I - A)^{-1}(I + A). Cayley
      CANNOT represent rotations with eigenvalue -1 (rotation angle = pi): those map
      to infinity, so a pi-rotation target is literally unrepresentable under Cayley.
      The discriminating holonomy in these tasks IS a pi-flip, so with Cayley the
      target transport was unreachable -> transport error floored, label stuck at
      chance. The matrix exponential expm(A) covers all of SO(d), including angle pi.

    (Historical note: this function used to be named `cayley`. It never actually
    computed a Cayley transform after the fix; it was renamed to `so_exp` so the name
    matches the math. A backward-compatible alias `cayley = so_exp` is provided below
    for any old call sites, but new code should call `so_exp`.)

    Args:
        A: (..., d, d) skew-symmetric
    Returns:
        Q: (..., d, d), orthogonal, det +1 (full SO(d) coverage)
    """
    return torch.matrix_exp(A)


# Backward-compatible alias. Deprecated: prefer so_exp.
cayley = so_exp


def build_ring_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Local ring mask. Token i can attend to:
      - itself
      - left neighbor
      - right neighbor

    Returns:
        mask: (seq_len, seq_len) bool
              True means allowed, False means blocked
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        mask[i, i] = True
        mask[i, (i - 1) % seq_len] = True
        mask[i, (i + 1) % seq_len] = True
    return mask