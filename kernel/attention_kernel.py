"""
attention_kernel.py -- SELF-CONTAINED, drop-in kernel-backed holo attention.

Everything in one file: Triton forward kernel, launcher, PyTorch recompute
forward, autograd.Function, and the drop-in nn.Module.

USAGE in any experiment:
    from attention_kernel import PathOrderedGeometricMHA
(identical interface to the original PathOrderedGeometricMHA)

KERNEL PATH is used ONLY when: mode="causal", attn_mask is exactly tril(L),
dropout is 0 (or eval mode), input is on CUDA, return_geometry=False.
Every other configuration falls back to attention_fast (exact full-grid
implementation, verified in all modes and masks; returns full geometry).

REQUIREMENTS (same folder): utils.py (so_exp, skew_from_vec),
                            attention_fast_chunked.py (reference + fallback).

SELF-TEST:  python attention_kernel.py   -> expect all OK / True.
"""
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

import os as _os, sys as _sys
# repo root (parent of kernel/) holds utils.py and attention.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)

from attention_fast_chunked import PathOrderedGeometricMHA as _RefMHA  # geometry (_geometry)
from attention_fast import PathOrderedGeometricMHA as _ExactMHA       # exact fallback, all modes


# =====================================================================
# 1. Triton forward kernel
# =====================================================================
@triton.jit
def fwd_kernel(
    Q_ptr, K_ptr, V_ptr, P_ptr, Hol_ptr, Out_ptr,
    L, d: tl.constexpr, scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    q_start = pid_q * BLOCK_Q
    q_off = q_start + tl.arange(0, BLOCK_Q)
    d_off = tl.arange(0, d)
    e_off = tl.arange(0, d)

    base = pid_bh * L * d
    base_P = pid_bh * L * d * d
    base_H = pid_bh * d * d

    q_mask = q_off < L

    q = tl.load(Q_ptr + base + q_off[:, None] * d + d_off[None, :],
                mask=q_mask[:, None], other=0.0)                       # (BQ,d)
    Pi = tl.load(P_ptr + base_P + q_off[:, None, None] * d * d
                 + d_off[None, :, None] * d + e_off[None, None, :],
                 mask=q_mask[:, None, None], other=0.0)                # (BQ,d,d)

    qh = tl.sum(Pi * q[:, :, None], axis=1)                            # (BQ,d)  P_i^T q_i

    Hol = tl.load(Hol_ptr + base_H + d_off[:, None] * d + e_off[None, :])  # (d,d)

    m_i = tl.full((BLOCK_Q,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, d), dtype=tl.float32)

    for k_start in range(0, L, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < L

        k = tl.load(K_ptr + base + k_off[:, None] * d + d_off[None, :],
                    mask=k_mask[:, None], other=0.0)                   # (BK,d)
        v = tl.load(V_ptr + base + k_off[:, None] * d + d_off[None, :],
                    mask=k_mask[:, None], other=0.0)
        Pj = tl.load(P_ptr + base_P + k_off[:, None, None] * d * d
                     + d_off[None, :, None] * d + e_off[None, None, :],
                     mask=k_mask[:, None, None], other=0.0)            # (BK,d,d)

        kh = tl.sum(Pj * k[:, :, None], axis=1)                        # (BK,d)
        vh = tl.sum(Pj * v[:, :, None], axis=1)

        kh_w = tl.sum(Hol[None, :, :] * kh[:, None, :], axis=2)        # Hol @ kh
        vh_w = tl.sum(Hol[None, :, :] * vh[:, None, :], axis=2)

        s_base = tl.sum(qh[:, None, :] * kh[None, :, :], axis=2) * scale
        s_wrap = tl.sum(qh[:, None, :] * kh_w[None, :, :], axis=2) * scale

        i_ge_j = q_off[:, None] >= k_off[None, :]
        s = tl.where(i_ge_j, s_base, s_wrap)

        allowed = (k_off[None, :] <= q_off[:, None]) & k_mask[None, :] & q_mask[:, None]
        s = tl.where(allowed, s, -1e30)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        corr = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * corr + tl.sum(p, axis=1)

        pv_base = tl.sum((p * i_ge_j)[:, :, None] * vh[None, :, :], axis=1)
        pv_wrap = tl.sum((p * (i_ge_j == 0))[:, :, None] * vh_w[None, :, :], axis=1)
        acc = acc * corr[:, None] + pv_base + pv_wrap
        m_i = m_new

    o = acc / l_i[:, None]
    out = tl.sum(Pi * o[:, None, :], axis=2)                           # P_i @ o

    tl.store(Out_ptr + base + q_off[:, None] * d + d_off[None, :], out,
             mask=q_mask[:, None])


# =====================================================================
# 2. Launcher
# =====================================================================
def run_fwd_kernel(q, k, v, P, Hol, scale):
    B, H, L, d = q.shape
    BH = B * H
    qf = q.reshape(BH, L, d).contiguous()
    kf = k.reshape(BH, L, d).contiguous()
    vf = v.reshape(BH, L, d).contiguous()
    Pf = P.reshape(BH, L, d, d).contiguous()
    Hf = Hol.reshape(BH, d, d).contiguous()
    out = torch.empty_like(qf)
    BLOCK_Q, BLOCK_K = 16, 16
    grid = (BH, triton.cdiv(L, BLOCK_Q))
    fwd_kernel[grid](qf, kf, vf, Pf, Hf, out, L, d, scale, BLOCK_Q, BLOCK_K)
    return out.reshape(B, H, L, d)


# =====================================================================
# 3. PyTorch recompute forward (for the autograd backward)
# =====================================================================
def _pytorch_forward(q, k, v, P, Hol, scale):
    B, H, L, d = q.shape
    device, dtype = q.device, q.dtype
    qh = torch.einsum("bhide,bhid->bhie", P, q)
    kh = torch.einsum("bhjde,bhjd->bhje", P, k)
    vh = torch.einsum("bhjde,bhjd->bhje", P, v)
    kh_w = torch.einsum("bhef,bhje->bhjf", Hol, kh)
    vh_w = torch.einsum("bhef,bhje->bhjf", Hol, vh)
    s_base = torch.einsum("bhie,bhje->bhij", qh, kh) * scale
    s_wrap = torch.einsum("bhie,bhje->bhij", qh, kh_w) * scale
    idx = torch.arange(L, device=device)
    i_ge_j = (idx[:, None] >= idx[None, :])
    s = torch.where(i_ge_j.view(1, 1, L, L), s_base, s_wrap)
    allowed = (idx[None, :] <= idx[:, None]).view(1, 1, L, L)
    s = s.masked_fill(~allowed, torch.finfo(dtype).min)
    attn = torch.softmax(s, dim=-1)
    ge = i_ge_j.view(1, 1, L, L).to(dtype)
    o_base = torch.einsum("bhij,bhje->bhie", attn * ge, vh)
    o_wrap = torch.einsum("bhij,bhje->bhie", attn * (1 - ge), vh_w)
    o = o_base + o_wrap
    return torch.einsum("bhide,bhie->bhid", P, o)


# =====================================================================
# 4. autograd.Function : kernel forward, recompute backward
# =====================================================================
class HoloAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, P, Hol, scale):
        out = run_fwd_kernel(q, k, v, P, Hol, scale)
        ctx.save_for_backward(q, k, v, P, Hol)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, P, Hol = ctx.saved_tensors
        scale = ctx.scale
        qd = q.detach().requires_grad_(True)
        kd = k.detach().requires_grad_(True)
        vd = v.detach().requires_grad_(True)
        Pd = P.detach().requires_grad_(True)
        Hd = Hol.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _pytorch_forward(qd, kd, vd, Pd, Hd, scale)
        grads = torch.autograd.grad(out, (qd, kd, vd, Pd, Hd),
                                    grad_outputs=grad_out.contiguous())
        return (*grads, None)


# =====================================================================
# 5. Drop-in module
# =====================================================================
def _is_exact_tril(mask, L):
    """The kernel applies causal masking internally and ignores mask contents,
    so it is only correct when the requested mask is exactly tril(L)."""
    if mask is None or tuple(mask.shape) != (L, L):
        return False
    tril = torch.ones(L, L, dtype=torch.bool, device=mask.device).tril()
    return torch.equal(mask.bool(), tril)


class PathOrderedGeometricMHA(_RefMHA):
    def _kernel_supported(self, x, attn_mask, return_geometry):
        return (self.mode == "causal"
                and not return_geometry
                and x.is_cuda
                and (self.dropout.p == 0.0 or not self.training)
                and _is_exact_tril(attn_mask, x.shape[1]))

    def forward(self, x, attn_mask=None, return_geometry=False):
        if not self._kernel_supported(x, attn_mask, return_geometry):
            return _ExactMHA.forward(self, x, attn_mask=attn_mask,
                                     return_geometry=return_geometry)
        B, L, D = x.shape
        q, k, v, P, Hol = self._geometry(x)
        out_heads = HoloAttnFn.apply(q, k, v, P, Hol, self.scale)
        out = out_heads.transpose(1, 2).reshape(B, L, D)
        return self.out(out), None


# =====================================================================
# 6. Self-tests
# =====================================================================
def _selftest():
    from attention import PathOrderedGeometricMHA as _OrigMHA   # original, repo root
    from utils import build_ring_attention_mask
    torch.manual_seed(0)
    B, L, D, Hh = 2, 12, 32, 4
    x = torch.randn(B, L, D, device="cuda")
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device="cuda"))
    ok_all = True

    # (a) kernel path vs chunked reference and vs the original implementation
    ref = _RefMHA(D, n_heads=Hh, mode="causal", edge_scale=1.5).cuda().eval()
    ker = PathOrderedGeometricMHA(D, n_heads=Hh, mode="causal", edge_scale=1.5).cuda().eval()
    orig = _OrigMHA(D, n_heads=Hh, mode="causal", edge_scale=1.5).cuda().eval()
    ker.load_state_dict(ref.state_dict()); orig.load_state_dict(ref.state_dict())
    assert ker._kernel_supported(x, tril, False), "kernel path not taken"
    with torch.no_grad():
        y_ref, _ = ref(x, attn_mask=tril)
        y_ker, _ = ker(x, attn_mask=tril)
        y_orig, _ = orig(x, attn_mask=tril)
    for name, a in (("vs chunked ref", y_ref), ("vs original attention.py", y_orig)):
        dy = (y_ker - a).abs().max().item(); ok = dy < 1e-4; ok_all &= ok
        print(f"[kernel path, causal tril] {name:<26} max|dy| = {dy:.2e}  {'OK' if ok else 'FAIL'}")

    # (b) fallback paths vs the original implementation
    ring = build_ring_attention_mask(L, x.device)
    for desc, mode, msk in (("inert, tril", "inert", tril),
                            ("causal, no mask", "causal", None),
                            ("causal, ring mask", "causal", ring)):
        k2 = PathOrderedGeometricMHA(D, n_heads=Hh, mode=mode, edge_scale=1.5).cuda().eval()
        o2 = _OrigMHA(D, n_heads=Hh, mode=mode, edge_scale=1.5).cuda().eval()
        o2.load_state_dict(k2.state_dict())
        with torch.no_grad():
            y1, _ = k2(x, attn_mask=msk)
            y2, _ = o2(x, attn_mask=msk)
        dy = (y1 - y2).abs().max().item(); ok = dy < 1e-4; ok_all &= ok
        print(f"[fallback, {desc:<17}] vs original attention.py  max|dy| = {dy:.2e}  {'OK' if ok else 'FAIL'}")

    # (c) backward math (float64 gradcheck of the recompute forward)
    B2, H2, L2, d2 = 1, 2, 6, 4
    sc = 1.0 / math.sqrt(d2); dt = torch.float64
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=dt, requires_grad=True)
    gc = torch.autograd.gradcheck(
        lambda q, k, v, P, H: _pytorch_forward(q, k, v, P, H, sc),
        (mk(B2, H2, L2, d2), mk(B2, H2, L2, d2), mk(B2, H2, L2, d2),
         mk(B2, H2, L2, d2, d2), mk(B2, H2, d2, d2)),
        eps=1e-6, atol=1e-4, rtol=1e-3)
    ok_all &= bool(gc)
    print(f"[gradcheck] recompute-backward math: {gc}")

    # (d) end-to-end backward through the kernel path
    x2 = torch.randn(B, L, D, device="cuda", requires_grad=True)
    y, _ = ker(x2, attn_mask=tril)
    y.sum().backward()
    fin = bool(torch.isfinite(x2.grad).all()); ok_all &= fin
    print(f"[backward] runs, grad finite: {fin}")
    print("\nALL OK" if ok_all else "\nSOME CHECKS FAILED -- do not trust results")


if __name__ == "__main__":
    _selftest()
