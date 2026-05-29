# -*- coding: utf-8 -*-
"""
cross_attention.py
------------------
Bidirectional Cross-Attention fusion module.
6 attention pairs (M1↔M2, M1↔M3, M2↔M3, M1↔M45, M2↔M45, M3↔M45)
h=8 heads, d_k=32, shared proj_dim=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """
    Single bidirectional cross-attention:
      CrossAttn(A→B) and CrossAttn(B→A) in parallel,
      outputs residual-added & layer-normed representations.
    """
    def __init__(self, dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = dim // n_heads

        # A→B
        self.W_qa = nn.Linear(dim, dim)
        self.W_ka = nn.Linear(dim, dim)
        self.W_va = nn.Linear(dim, dim)
        self.out_a = nn.Linear(dim, dim)

        # B→A
        self.W_qb = nn.Linear(dim, dim)
        self.W_kb = nn.Linear(dim, dim)
        self.W_vb = nn.Linear(dim, dim)
        self.out_b = nn.Linear(dim, dim)

        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def _attend(self, Q, K, V, W_q, W_k, W_v, W_out):
        """Scaled dot-product attention (batch, dim) → (batch, dim)"""
        B, D = Q.shape
        # Expand to (B, 1, D) for multi-head split
        q = W_q(Q).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,1,d_k)
        k = W_k(K).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        v = W_v(V).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,1,1)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.matmul(attn, v)                                         # (B,h,1,d_k)
        out    = out.transpose(1, 2).contiguous().view(B, D)
        return W_out(out), attn.squeeze(-1).squeeze(-1)  # (B,D), (B,h)

    def forward(self, z_a, z_b):
        """
        Args:
            z_a, z_b: (batch, dim) projected modality representations
        Returns:
            z_a', z_b': residual-updated representations
            attn_ab, attn_ba: attention weights (B, n_heads)
        """
        # A attends to B
        attn_out_a, attn_ab = self._attend(z_a, z_b, z_b,
                                            self.W_qa, self.W_ka, self.W_va, self.out_a)
        z_a_new = self.norm_a(z_a + self.drop(attn_out_a))

        # B attends to A
        attn_out_b, attn_ba = self._attend(z_b, z_a, z_a,
                                            self.W_qb, self.W_kb, self.W_vb, self.out_b)
        z_b_new = self.norm_b(z_b + self.drop(attn_out_b))

        return z_a_new, z_b_new, attn_ab, attn_ba


class FusionFFN(nn.Module):
    """
    Feed-Forward Network after attention fusion:
    concatenate 6 cross-attended representations → 1536 → 512 → 256
    """
    def __init__(self, n_mods=4, proj_dim=256, hidden=512, out_dim=256, dropout=0.2):
        super().__init__()
        in_dim = n_mods * proj_dim  # 4 × 256 = 1024 (M1, M2, M3, M45_combined)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, *zs):
        cat = torch.cat(zs, dim=-1)
        return self.norm(self.net(cat))


class CrossAttentionFusion(nn.Module):
    """
    Full bidirectional cross-attention fusion for M1, M2, M3, M45.
    6 attention pairs applied:
      M1↔M2, M1↔M3, M1↔M45, M2↔M3, M2↔M45, M3↔M45
    Then pooled representations fed into FusionFFN → z_fusion (256-d).
    """
    def __init__(self, proj_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_12  = CrossAttentionBlock(proj_dim, n_heads, dropout)
        self.attn_13  = CrossAttentionBlock(proj_dim, n_heads, dropout)
        self.attn_145 = CrossAttentionBlock(proj_dim, n_heads, dropout)
        self.attn_23  = CrossAttentionBlock(proj_dim, n_heads, dropout)
        self.attn_245 = CrossAttentionBlock(proj_dim, n_heads, dropout)
        self.attn_345 = CrossAttentionBlock(proj_dim, n_heads, dropout)

        # After 6 pairs each modality has been updated multiple times → average pool
        self.ffn = FusionFFN(n_mods=4, proj_dim=proj_dim,
                              hidden=512, out_dim=proj_dim, dropout=dropout)

    def forward(self, z1, z2, z3, z45):
        """
        Args:
            z1, z2, z3, z45 : (B, 256) projected modality representations
        Returns:
            z_fusion : (B, 256)
            attn_weights : dict of attention weight tensors for XAI
        """
        attn_w = {}

        # Pair M1-M2
        z1_12, z2_12, aw12ab, aw12ba = self.attn_12(z1, z2)
        attn_w['M1→M2'] = aw12ab; attn_w['M2→M1'] = aw12ba

        # Pair M1-M3
        z1_13, z3_13, aw13ab, aw13ba = self.attn_13(z1_12, z3)
        attn_w['M1→M3'] = aw13ab; attn_w['M3→M1'] = aw13ba

        # Pair M1-M45
        z1_f, z45_1, aw145ab, aw145ba = self.attn_145(z1_13, z45)
        attn_w['M1→M45'] = aw145ab; attn_w['M45→M1'] = aw145ba

        # Pair M2-M3
        z2_23, z3_23, aw23ab, aw23ba = self.attn_23(z2_12, z3_13)
        attn_w['M2→M3'] = aw23ab; attn_w['M3→M2'] = aw23ba

        # Pair M2-M45
        z2_f, z45_2, aw245ab, aw245ba = self.attn_245(z2_23, z45_1)
        attn_w['M2→M45'] = aw245ab; attn_w['M45→M2'] = aw245ba

        # Pair M3-M45
        z3_f, z45_f, aw345ab, aw345ba = self.attn_345(z3_23, z45_2)
        attn_w['M3→M45'] = aw345ab; attn_w['M45→M3'] = aw345ba

        # Fuse all updated representations
        z_fusion = self.ffn(z1_f, z2_f, z3_f, z45_f)

        return z_fusion, attn_w
