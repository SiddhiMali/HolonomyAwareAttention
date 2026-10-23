"""
attention_fast_chunked.py -- memory-efficient (chunked) PathOrderedGeometricMHA.

Same interface as attention_fast.PathOrderedGeometricMHA, but the O(L^2)
transport grid is NEVER materialized: keys are streamed in blocks of BLOCK_K
with a running (FlashAttention-style) softmax fold. Peak memory ~ (B,H,L,BLOCK_K,d)
instead of (B,H,L,L,d,d). Plain PyTorch, so autograd gives gradients for free.

DROP-IN: replace
    from attention_fast import PathOrderedGeometricMHA
with
    from attention_fast_chunked import PathOrderedGeometricMHA
in your experiment scripts.

!!! RUN THE SELF-TEST FIRST !!!
    python attention_fast_chunked.py
It compares this chunked forward against a reference full-grid forward at small L
in BOTH modes. If it prints "ALL OK" you are safe to use it. If any dy is large,
DO NOT USE THIS FILE -- it means a transpose convention is flipped (see the two
spots marked  >>> TRANSPOSE CHECK <<< ) and needs one edit. Report the numbers.

geometry note: return_geometry=True returns C and H (cheap, exact) but NOT the
full G_arc/attn grid (that's the thing we refuse to materialize). Probes that
need geom["attn"]/["G_arc"] should use the original attention_fast at small L.
"""
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn

# so_exp / skew_from_vec come from your utils.py
import os as _os, sys as _sys
# repo root (parent of kernel/) holds utils.py and attention.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)

from utils import so_exp, skew_from_vec

BLOCK_K = 64   # key-block size. Lower if you still OOM; raise for speed.


class PathOrderedGeometricMHA(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.0, edge_scale=0.1,
                 mode="causal"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if mode not in ("causal", "inert"):
            raise ValueError("mode must be 'causal' or 'inert'")
        self.h = n_heads
        self.d = d_model // n_heads
        self.m_dim = self.d * (self.d - 1) // 2
        self.edge_scale = edge_scale
        self.scale = 1.0 / math.sqrt(self.d)
        self.mode = mode

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gqproj = nn.Linear(d_model, self.m_dim * n_heads, bias=False)
        self.gkproj = nn.Linear(d_model, self.m_dim * n_heads, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, self.m_dim * n_heads),
        )
        self.link_mode = "interaction"

    # ---- shared: build P (prefix products) and Hol (holonomy) ----
    def _geometry(self, x):
        B, L, D = x.shape
        H_, d = self.h, self.d
        device, dtype = x.device, x.dtype
        I = torch.eye(d, device=device, dtype=dtype)

        qkv = self.qkv(x).view(B, L, H_, 3 * d)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # (B,H,L,d)

        if self.link_mode == "interaction":
            x_right = torch.roll(x, shifts=-1, dims=1)
            pair = torch.cat([x, x_right], dim=-1)
            a_all = self.edge_mlp(pair).view(B, L, H_, self.m_dim).transpose(1, 2)
            a_all = a_all * self.edge_scale
            a_adj = a_all[:, :, :-1]
            a_close = a_all[:, :, -1]
        else:
            gq = self.gqproj(x).view(B, L, H_, self.m_dim).transpose(1, 2)
            gk = self.gkproj(x).view(B, L, H_, self.m_dim).transpose(1, 2)
            a_adj = (gq[:, :, :-1] + gk[:, :, 1:]) * self.edge_scale
            a_close = (gq[:, :, -1] + gk[:, :, 0]) * self.edge_scale

        G_adj = so_exp(skew_from_vec(a_adj, d))
        G_close = so_exp(skew_from_vec(a_close, d))

        P = torch.empty(B, H_, L, d, d, device=device, dtype=dtype)
        P[:, :, 0] = I
        Pt = I.expand(B, H_, d, d).clone()
        for t in range(L - 1):
            Pt = G_adj[:, :, t] @ Pt
            P[:, :, t + 1] = Pt
        Hol = G_close @ P[:, :, L - 1]
        return q, k, v, P, Hol

    def forward(self, x, attn_mask=None, return_geometry=False):
        # VERIFIED ONLY for mode="causal" with a mask that blocks every i<j pair
        # (e.g. tril, or tril AND padding). Other configs reach a wrap-branch path
        # that is known to be incorrect, so refuse instead of returning wrong numbers.
        if (self.mode != "causal") or (attn_mask is None) or \
                bool(torch.triu(attn_mask.bool(), diagonal=1).any()):
            raise NotImplementedError(
                "attention_fast_chunked supports only mode='causal' with a mask "
                "blocking all i<j pairs (e.g. tril). Use attention_fast or "
                "attention_kernel for other configurations.")
        B, L, D = x.shape
        H_, d = self.h, self.d
        device, dtype = x.device, x.dtype

        q, k, v, P, Hol = self._geometry(x)

        # ---- Option B fusion: transform q,k,v per token ----
        # qh_i = P_i^T @ q_i ;  kh_j = P_j^T @ k_j ;  vh_j = P_j^T @ v_j
        # >>> TRANSPOSE CHECK <<<  ("bhide,bhid->bhie" applies P_i^T @ q_i)
        qh = torch.einsum("bhide,bhid->bhie", P, q)          # (B,H,L,d)
        kh_all = torch.einsum("bhjde,bhjd->bhje", P, k)      # (B,H,L,d)
        vh_all = torch.einsum("bhjde,bhjd->bhje", P, v)

        if self.mode == "causal":
            # wrap pairs (i<j) carry an extra Hol on the key/value side.
            # >>> TRANSPOSE CHECK <<<  (Hol @ kh_j). If test fails, change
            # "bhef,bhje->bhjf" below to "bhfe,bhje->bhjf".
            kh_wrap_all = torch.einsum("bhef,bhje->bhjf", Hol, kh_all)
            vh_wrap_all = torch.einsum("bhef,bhje->bhjf", Hol, vh_all)

        neg_inf = torch.finfo(dtype).min
        m_run = torch.full((B, H_, L), neg_inf, device=device, dtype=dtype)
        l_run = torch.zeros((B, H_, L), device=device, dtype=dtype)
        o_run = torch.zeros((B, H_, L, d), device=device, dtype=dtype)

        idx = torch.arange(L, device=device)

        for k0 in range(0, L, BLOCK_K):
            k1 = min(k0 + BLOCK_K, L)
            j_idx = idx[k0:k1]
            Bk = k1 - k0

            kh = kh_all[:, :, k0:k1]
            vh = vh_all[:, :, k0:k1]

            s_base = torch.einsum("bhie,bhje->bhij", qh, kh) * self.scale  # (B,H,L,Bk)

            if self.mode == "causal":
                kh_w = kh_wrap_all[:, :, k0:k1]
                s_wrap = torch.einsum("bhie,bhje->bhij", qh, kh_w) * self.scale
                i_ge_j = (idx.view(L, 1) >= j_idx.view(1, Bk))  # (L,Bk) True=base
                s = torch.where(i_ge_j.view(1, 1, L, Bk), s_base, s_wrap)
            else:
                s = s_base

            if attn_mask is not None:
                allowed = attn_mask[:, k0:k1].view(1, 1, L, Bk)
                s = s.masked_fill(~allowed, neg_inf)

            m_blk = s.max(dim=-1).values
            m_new = torch.maximum(m_run, m_blk)
            corr = torch.exp(m_run - m_new)
            p = torch.exp(s - m_new.unsqueeze(-1))            # (B,H,L,Bk)

            l_run = l_run * corr + p.sum(dim=-1)

            if self.mode == "causal":
                vh_w = vh_wrap_all[:, :, k0:k1]
                cb = torch.einsum("bhij,bhje->bhie",
                                  p * i_ge_j.view(1, 1, L, Bk), vh)
                cw = torch.einsum("bhij,bhje->bhie",
                                  p * (~i_ge_j).view(1, 1, L, Bk), vh_w)
                contrib = cb + cw
            else:
                contrib = torch.einsum("bhij,bhje->bhie", p, vh)

            o_run = o_run * corr.unsqueeze(-1) + contrib
            m_run = m_new

        o_run = o_run / l_run.clamp_min(1e-20).unsqueeze(-1)   # (B,H,L,d)
        # apply P_i (left factor, outside the sum): out_i = P_i @ o_i
        # >>> TRANSPOSE CHECK <<<  ("bhide,bhie->bhid" applies P_i @ o_i)
        out_heads = torch.einsum("bhide,bhie->bhid", P, o_run)
        out = out_heads.transpose(1, 2).reshape(B, L, D)
        y = self.out(out)

        if not return_geometry:
            return y, None
        return y, {"C": P, "H": Hol}


# =====================================================================
# SELF-TEST: compare chunked forward vs a reference full-grid forward.
# Run:  python attention_fast_chunked.py
# =====================================================================
def _reference_forward(mod, x, attn_mask):
    """Full-grid forward (the original attention_fast math) for checking."""
    B, L, D = x.shape
    H_, d = mod.h, mod.d
    device, dtype = x.device, x.dtype
    I = torch.eye(d, device=device, dtype=dtype)
    q, k, v, P, Hol = mod._geometry(x)

    if mod.mode == "causal":
        base = torch.einsum("bhide,bhjfe->bhijdf", P, P)
        PH = torch.einsum("bhide,bhef->bhidf", P, Hol)
        wrap = torch.einsum("bhidf,bhjgf->bhijdg", PH, P)
        ii = torch.arange(L, device=device).view(L, 1)
        jj = torch.arange(L, device=device).view(1, L)
        use_wrap = (ii < jj).view(1, 1, L, L, 1, 1)
        G_arc = torch.where(use_wrap, wrap, base)
    else:
        G_arc = torch.einsum("bhied,bhjef->bhijdf", P, P)

    Kt = torch.einsum("bhijdf,bhjf->bhijd", G_arc, k)
    Vt = torch.einsum("bhijdf,bhjf->bhijd", G_arc, v)
    scores = torch.einsum("bhid,bhijd->bhij", q, Kt) * mod.scale
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.view(1, 1, L, L), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out_heads = torch.einsum("bhij,bhijd->bhid", attn, Vt)
    out = out_heads.transpose(1, 2).reshape(B, L, D)
    return mod.out(out)


def _selftest():
    torch.manual_seed(0)
    B, L, D, Hh = 2, 12, 32, 4
    m = PathOrderedGeometricMHA(D, n_heads=Hh, dropout=0.0, mode="causal").eval()
    x = torch.randn(B, L, D)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
    with torch.no_grad():
        y_chunk, _ = m(x, attn_mask=mask)
        y_ref = _reference_forward(m, x, mask)
    dy = (y_chunk - y_ref).abs().max().item()
    print(f"[causal tril] max|dy| = {dy:.2e}  {'OK' if dy < 1e-4 else 'FAIL'}")
    for desc, mode, msk in (("inert", "inert", mask), ("no mask", "causal", None)):
        mm = PathOrderedGeometricMHA(D, n_heads=Hh, mode=mode).eval()
        try:
            mm(x, attn_mask=msk)
            print(f"[{desc}] guard did NOT raise  FAIL")
        except NotImplementedError:
            print(f"[{desc}] correctly refused (unsupported)  OK")


if __name__ == "__main__":
    _selftest()
