from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn



from utils import so_exp
from utils import skew_from_vec

class PathOrderedGeometricMHA(nn.Module):
    """
    Path-ordered geometric multi-head attention over a ring/path.

    Main idea:
      - each head learns local link transports
      - prefix transports define endpoint-to-endpoint transport
      - q/k comparisons happen after transporting keys into query frame
      - full loop product yields a holonomy matrix

    Notes:
      - this version supports a structural attention mask
      - mask affects attention scores, not transport construction
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        edge_scale: float = 0.1,
        mode: str = "causal",     # "causal" = arc transport (holonomy load-bearing)
                                  # "inert"  = endpoint transport C_i^T C_j (telescopes)
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

        # link parameter projections (ADDITIVE form -- kept for mode="additive")
        self.gqproj = nn.Linear(d_model, self.m_dim * n_heads, bias=False)
        self.gkproj = nn.Linear(d_model, self.m_dim * n_heads, bias=False)

        # INTERACTION form: an MLP on the concatenated PAIR of adjacent token features.
        # The additive gq(x_i)+gk(x_j) form cannot represent arbitrary SO(d) targets
        # (confirmed: transport_err floors ~0.084). A joint MLP on (x_i, x_j) can express
        # an arbitrary function of the pair -> can hit arbitrary rotations.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, self.m_dim * n_heads),
        )
        # "interaction" (default now) uses edge_mlp; "additive" uses gqproj+gkproj.
        self.link_mode = "interaction"

    def forward(
        self,
        x: torch.Tensor,                        # (B, L, D)
        attn_mask: Optional[torch.Tensor] = None,  # (L, L) bool, True = allowed
        return_geometry: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, L, D = x.shape
        device, dtype = x.device, x.dtype

        I = torch.eye(self.d, device=device, dtype=dtype)

        # ----------------------------
        # QKV
        # ----------------------------
        qkv = self.qkv(x).view(B, L, self.h, 3 * self.d)
        q, k, v = qkv.chunk(3, dim=-1)   # (B, L, H, d)
        q = q.transpose(1, 2)            # (B, H, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----------------------------
        # Link parameters
        # ----------------------------
        if self.link_mode == "interaction":
            # edge t transport = MLP( concat(x_t, x_{t+1}) ), around the ring.
            # adjacent pairs (t, t+1) for t=0..L-2, plus closing pair (L-1, 0).
            x_left  = x                                  # (B,L,D)
            x_right = torch.roll(x, shifts=-1, dims=1)   # x_{t+1}, wraps last->first
            pair = torch.cat([x_left, x_right], dim=-1)  # (B,L,2D)
            a_all = self.edge_mlp(pair).view(B, L, self.h, self.m_dim).transpose(1, 2)
            a_all = a_all * self.edge_scale              # (B,H,L,m): edge t = (t -> t+1)
            a_adj   = a_all[:, :, :-1]                   # (B,H,L-1,m) edges 0..L-2
            a_close = a_all[:, :, -1]                    # (B,H,m) closing edge L-1 -> 0
        else:  # additive fallback: a_ij = gq(x_i) + gk(x_j)
            gq = self.gqproj(x).view(B, L, self.h, self.m_dim).transpose(1, 2)
            gk = self.gkproj(x).view(B, L, self.h, self.m_dim).transpose(1, 2)
            a_adj   = (gq[:, :, :-1] + gk[:, :, 1:]) * self.edge_scale
            a_close = (gq[:, :, -1] + gk[:, :, 0]) * self.edge_scale

        G_adj = so_exp(skew_from_vec(a_adj, self.d))                      # (B,H,L-1,d,d)
        G_close = so_exp(skew_from_vec(a_close, self.d))                  # (B,H,d,d)
        G_cycle = torch.cat([G_adj, G_close.unsqueeze(2)], dim=2)         # (B,H,L,d,d)

        # ----------------------------
        # Holonomy H = product around loop
        # ----------------------------
        H = I.expand(B, self.h, self.d, self.d).clone()
        for t in range(L):
            H = G_cycle[:, :, t] @ H

        # ----------------------------
        # Prefix transports C_t
        # C[:, :, t] = transport from token t frame to token 0 frame
        # ----------------------------
        C = torch.zeros(B, self.h, L, self.d, self.d, device=device, dtype=dtype)
        C[:, :, 0] = I

        Ct = I.expand(B, self.h, self.d, self.d).clone()
        for t in range(L - 1):
            Ct = G_adj[:, :, t] @ Ct
            C[:, :, t + 1] = Ct

        C_T = C.transpose(-2, -1)

        # ----------------------------
        # Streaming attention with transported keys/values
        # ----------------------------
        out_heads = torch.zeros(B, self.h, L, self.d, device=device, dtype=dtype)

        # default: allow all if no mask given
        if attn_mask is None:
            attn_mask = torch.ones(L, L, dtype=torch.bool, device=device)

        # ----------------------------
        # Transport construction, controlled by self.mode.
        #
        # "causal" (arc transport): transport j -> i by WALKING the ring arc,
        #   composing the actual edge transports G_cycle. The forward arc
        #   j -> j+1 -> ... -> i crosses G_close for wrap-around pairs, so the
        #   closing edge (and thus the holonomy) genuinely influences the output.
        #
        # "inert" (endpoint transport): G_{i<-j} = C_i^T C_j, built only from the
        #   prefix products of G_adj. This telescopes: G_close never enters the
        #   output, so the holonomy is causally inert. This is the ABLATION that
        #   isolates whether the win comes from loop composition (holonomy) or just
        #   from the extra rotation machinery (which both modes share identically).
        # ----------------------------
        G_arc = torch.zeros(B, self.h, L, L, self.d, self.d, device=device, dtype=dtype)
        if self.mode == "causal":
            for i in range(L):
                for j in range(L):
                    U = I.expand(B, self.h, self.d, self.d).clone()
                    t = j
                    while t != i:
                        U = G_cycle[:, :, t] @ U      # edge t maps frame t -> t+1
                        t = (t + 1) % L
                    G_arc[:, :, i, j] = U
        else:  # inert: endpoint transport C_i^T C_j (telescopes; G_close unused)
            for i in range(L):
                CiT = C_T[:, :, i]                                    # (B,H,d,d)
                G_arc[:, :, i] = torch.einsum("bhde,bhjef->bhjdf", CiT, C)

        # buffer of per-query attention rows, for the entropy/dilution probe
        attn_rows = torch.zeros(B, self.h, L, L, device=device, dtype=dtype)

        for i in range(L):
            # transport from token j frame to token i frame
            G_i = G_arc[:, :, i]  # (B,H,L,d,d)

            Kt = torch.einsum("bhjdf,bhjf->bhjd", G_i, k)    # transported keys
            Vt = torch.einsum("bhjdf,bhjf->bhjd", G_i, v)    # transported values

            qi = q[:, :, i]                                  # (B,H,d)
            scores = torch.einsum("bhd,bhjd->bhj", qi, Kt) * self.scale

            # mask blocked positions
            allowed = attn_mask[i].view(1, 1, L)            # (1,1,L)
            scores = scores.masked_fill(~allowed, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn_rows[:, :, i] = attn                        # store BEFORE dropout
            attn = self.dropout(attn)

            out_heads[:, :, i] = torch.einsum("bhj,bhjd->bhd", attn, Vt)

        out = out_heads.transpose(1, 2).reshape(B, L, D)
        y = self.out(out)

        if not return_geometry:
            return y, None

        geometry = {
            "G_adj": G_adj,   # (B,H,L-1,d,d)
            "G_close": G_close,
            "G_cycle": G_cycle,
            "C": C,           # (B,H,L,d,d)
            "H": H,           # (B,H,d,d)
            "attn": attn_rows,  # (B,H,L,L) per-query attention rows (post-softmax, pre-dropout)
        }
        return y, geometry
