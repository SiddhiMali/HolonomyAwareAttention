from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn

import os as _os, sys as _sys
# repo root (parent of kernel/) holds utils.py and attention.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.append(_ROOT)

from utils import so_exp
from utils import skew_from_vec


class PathOrderedGeometricMHA(nn.Module):
    """
    Vectorized path-ordered geometric multi-head attention over a ring/path.

    Semantics are identical to the reference implementation; the two Python
    loops that built G_arc (O(L^2) iterations, O(L^3) matmuls) and the
    streaming attention loop (O(L) iterations) have been removed.

    Key identity used
    -----------------
    Let P_t be the prefix product of edge transports mapping frame 0 -> frame t,
    with P_0 = I and P_{t} = G_adj[t-1] @ ... @ G_adj[0]. Note C[:, :, t] == P_t.
    Let H be the full holonomy (product of all L edges incl. the closing edge).

    The forward arc transport j -> i (walking the ring forward) is:
        i >= j :  P_i @ P_j^T              (no wrap; diagonal i==j gives I)
        i <  j :  P_i @ H @ P_j^T          (wraps once, picks up holonomy)

    Because every transport is in SO(d), P_j^{-1} == P_j^T. This reproduces the
    reference's per-pair arc walk exactly (verified to machine precision), and
    reduces it to batched einsums.

    The "inert" mode instead uses the reference's endpoint transport
    G[i,j] = C_i^T @ C_j = P_i^T @ P_j (telescopes; closing edge / holonomy
    never enters the output).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        edge_scale: float = 0.1,
        mode: str = "causal",
    ):
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

    def forward(
        self,
        x: torch.Tensor,                           # (B, L, D)
        attn_mask: Optional[torch.Tensor] = None,  # (L, L) bool, True = allowed
        return_geometry: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, L, D = x.shape
        device, dtype = x.device, x.dtype
        H_, d = self.h, self.d
        I = torch.eye(d, device=device, dtype=dtype)

        # ---- QKV -----------------------------------------------------------
        qkv = self.qkv(x).view(B, L, H_, 3 * d)
        q, k, v = qkv.chunk(3, dim=-1)             # (B,L,H,d)
        q = q.transpose(1, 2)                      # (B,H,L,d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ---- Link parameters ----------------------------------------------
        if self.link_mode == "interaction":
            x_right = torch.roll(x, shifts=-1, dims=1)
            pair = torch.cat([x, x_right], dim=-1)                    # (B,L,2D)
            a_all = self.edge_mlp(pair).view(B, L, H_, self.m_dim).transpose(1, 2)
            a_all = a_all * self.edge_scale                          # (B,H,L,m)
            a_adj = a_all[:, :, :-1]
            a_close = a_all[:, :, -1]
        else:
            gq = self.gqproj(x).view(B, L, H_, self.m_dim).transpose(1, 2)
            gk = self.gkproj(x).view(B, L, H_, self.m_dim).transpose(1, 2)
            a_adj = (gq[:, :, :-1] + gk[:, :, 1:]) * self.edge_scale
            a_close = (gq[:, :, -1] + gk[:, :, 0]) * self.edge_scale

        G_adj = so_exp(skew_from_vec(a_adj, d))                       # (B,H,L-1,d,d)
        G_close = so_exp(skew_from_vec(a_close, d))                   # (B,H,d,d)
        G_cycle = torch.cat([G_adj, G_close.unsqueeze(2)], dim=2)     # (B,H,L,d,d)

        # ---- Prefix products P_t (== C) via cumulative left-multiply -------
        # P[t] = G_adj[t-1] @ ... @ G_adj[0], P[0] = I.
        # Sequential in L (O(L) matmuls) but batched over B,H; this is the
        # only O(L) sweep left and it is cheap relative to the old O(L^3) walk.
        P = torch.empty(B, H_, L, d, d, device=device, dtype=dtype)
        P[:, :, 0] = I
        Pt = I.expand(B, H_, d, d).clone()
        for t in range(L - 1):
            Pt = G_adj[:, :, t] @ Pt
            P[:, :, t + 1] = Pt
        C = P                                                        # alias

        # Holonomy H = G_close @ P_{L-1}  (all L edges)
        Hol = G_close @ P[:, :, L - 1]                               # (B,H,d,d)

        # ---- Vectorized arc transport G_arc[i,j] --------------------------
        if self.mode == "causal":
            # arc[i,j] = P_i @ P_j^T for i>=j, and P_i @ H @ P_j^T for i<j.
            # The strict i<j branch is where the forward walk wraps around and
            # crosses the closing edge -> holonomy enters. Diagonal (i==j) sits
            # in the no-wrap branch and gives identity, matching the reference.
            base = torch.einsum("bhide,bhjfe->bhijdf", P, P)       # P_i @ P_j^T
            PH = torch.einsum("bhide,bhef->bhidf", P, Hol)         # P_i @ H
            wrap = torch.einsum("bhidf,bhjgf->bhijdg", PH, P)      # P_i @ H @ P_j^T
            ii = torch.arange(L, device=device).view(L, 1)
            jj = torch.arange(L, device=device).view(1, L)
            use_wrap = (ii < jj).view(1, 1, L, L, 1, 1)            # strict <
            G_arc = torch.where(use_wrap, wrap, base)
        else:
            # inert: endpoint transport C_i^T @ C_j = P_i^T @ P_j
            G_arc = torch.einsum("bhied,bhjef->bhijdf", P, P)      # (B,H,L,L,d,d)

        # ---- Transported keys/values, all queries at once -----------------
        # Kt[i,j] = G_arc[i,j] @ k[j];  Vt[i,j] = G_arc[i,j] @ v[j]
        Kt = torch.einsum("bhijdf,bhjf->bhijd", G_arc, k)          # (B,H,L,L,d)
        Vt = torch.einsum("bhijdf,bhjf->bhijd", G_arc, v)

        # scores[i,j] = <q_i, Kt[i,j]>
        scores = torch.einsum("bhid,bhijd->bhij", q, Kt) * self.scale

        if attn_mask is None:
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=device)
        scores = scores.masked_fill(~attn_mask.view(1, 1, L, L), float("-inf"))

        attn = torch.softmax(scores, dim=-1)                        # (B,H,L,L)
        attn_rows = attn
        attn = self.dropout(attn)

        out_heads = torch.einsum("bhij,bhijd->bhid", attn, Vt)      # (B,H,L,d)
        out = out_heads.transpose(1, 2).reshape(B, L, D)
        y = self.out(out)

        if not return_geometry:
            return y, None

        geometry = {
            "G_adj": G_adj,
            "G_close": G_close,
            "G_cycle": G_cycle,
            "C": C,
            "H": Hol,
            "attn": attn_rows,
        }
        return y, geometry
