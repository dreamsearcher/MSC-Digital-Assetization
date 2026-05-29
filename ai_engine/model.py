# -*- coding: utf-8 -*-
"""
model.py
--------
MDAFModel: Full multimodal AI engine
  - M1~M5 encoders → projection → Cross-Attention fusion
  - Dual output heads: MQS regression + 5-class grade classification
  - Phase-2 tau_mean inverse estimation module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (MorphEncoder, MetabolicEncoder, DonorEncoder,
                       FlowCytometryEncoder, ManufacturingEncoder,
                       ModalityProjection)
from .cross_attention import CrossAttentionFusion


PROJ_DIM  = 256
N_GRADES  = 5    # S/A/B/C/D


class MDAFModel(nn.Module):
    """
    Full MSC Digital Assetization Framework AI Engine.

    Architecture:
      M1 (morph 6-d + CNN 512-d) → MorphEncoder → 512-d → Proj → 256-d
      M2 (3-d)                   → MetabolicEncoder → 128-d → Proj → 256-d
      M3 (15-d)                  → DonorEncoder → 256-d → Proj → 256-d
      M4 (6-d) + M5 (5-d)       → FlowEnc/ManufEnc → 64+64=128-d → Proj → 256-d
      Bidirectional Cross-Attention × 6 pairs (h=8) → FusionFFN → z_fusion (256-d)
      Regression head  → MQS ∈ [0,100]
      Classification head → grade logits (5-class)
      Phase-2 inverse → tau_mean estimate from z_fusion
    """

    def __init__(self, dropout=0.2):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────
        self.enc_m1  = MorphEncoder(morph_dim=6, cnn_dim=512,
                                     out_dim=512, dropout=dropout)
        self.enc_m2  = MetabolicEncoder(in_dim=3,  out_dim=128, dropout=dropout)
        self.enc_m3  = DonorEncoder(in_dim=15, out_dim=256, dropout=dropout)
        self.enc_m4  = FlowCytometryEncoder(in_dim=6,  out_dim=64, dropout=dropout*0.5)
        self.enc_m5  = ManufacturingEncoder(in_dim=5,  out_dim=64, dropout=dropout*0.5)

        # ── Projection to shared 256-d space ──────────────────
        self.proj_m1  = ModalityProjection(512, PROJ_DIM)
        self.proj_m2  = ModalityProjection(128, PROJ_DIM)
        self.proj_m3  = ModalityProjection(256, PROJ_DIM)
        self.proj_m45 = ModalityProjection(128, PROJ_DIM)   # M4+M5 concatenated

        # ── Cross-Attention Fusion ────────────────────────────
        self.fusion = CrossAttentionFusion(proj_dim=PROJ_DIM, n_heads=8,
                                           dropout=dropout)

        # ── Output heads ──────────────────────────────────────
        # MQS regression: 256→64→1, sigmoid×100
        self.reg_head = nn.Sequential(
            nn.Linear(PROJ_DIM, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Grade classification: 256→64→5, softmax (applied in loss)
        self.cls_head = nn.Sequential(
            nn.Linear(PROJ_DIM, 64),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, N_GRADES),
        )

        # Phase-2: tau_mean inverse estimation (z_fusion → tau_mean)
        self.tau_inv_head = nn.Sequential(
            nn.Linear(PROJ_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # scaled to [1.5, 4.0] in forward
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        """
        Args:
            batch: dict with keys m1_morph, m1_cnn, m2, m3, m4, m5
        Returns:
            mqs       : (B,) scaled [0,100]
            grade_logits : (B, 5)
            tau_pred  : (B,) estimated tau_mean [1.5, 4.0]
            z_fusion  : (B, 256) batch embedding for digital twin
            attn_weights : dict of attention weight tensors
        """
        # Encode
        z_m1  = self.enc_m1(batch['m1_morph'], batch['m1_cnn'])   # (B,512)
        z_m2  = self.enc_m2(batch['m2'])                           # (B,128)
        z_m3  = self.enc_m3(batch['m3'])                           # (B,256)
        z_m4  = self.enc_m4(batch['m4'])                           # (B,64)
        z_m5  = self.enc_m5(batch['m5'])                           # (B,64)
        z_m45 = torch.cat([z_m4, z_m5], dim=-1)                   # (B,128)

        # Project to shared space
        p1  = self.proj_m1(z_m1)
        p2  = self.proj_m2(z_m2)
        p3  = self.proj_m3(z_m3)
        p45 = self.proj_m45(z_m45)

        # Cross-Attention fusion
        z_fusion, attn_w = self.fusion(p1, p2, p3, p45)

        # Output heads
        mqs          = self.reg_head(z_fusion).squeeze(-1) * 100.0   # [0,100]
        grade_logits = self.cls_head(z_fusion)                        # (B,5)
        tau_pred     = self.tau_inv_head(z_fusion).squeeze(-1) * 2.5 + 1.5  # [1.5,4.0]

        return mqs, grade_logits, tau_pred, z_fusion, attn_w


class MDAFLoss(nn.Module):
    """
    Combined loss:
      L = 0.6 × MSE(MQS) + 0.4 × CrossEntropy(grade) + 0.05 × MSE(tau_inv)
    """
    def __init__(self, alpha=0.6, beta=0.4, gamma=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.mse   = nn.MSELoss()
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, mqs_pred, grade_logits, tau_pred,
                mqs_true, grade_true, tau_true=None):
        loss_mqs   = self.mse(mqs_pred, mqs_true)
        loss_grade = self.ce(grade_logits, grade_true)
        loss = self.alpha * loss_mqs + self.beta * loss_grade

        if tau_true is not None:
            loss_tau = self.mse(tau_pred, tau_true)
            loss += self.gamma * loss_tau

        return loss, loss_mqs, loss_grade


def mqs_to_grade_tensor(mqs):
    """Convert MQS scalar tensor to grade index tensor."""
    grade = torch.zeros_like(mqs, dtype=torch.long)
    grade[mqs >= 90] = 0  # S
    grade[(mqs >= 80) & (mqs < 90)] = 1  # A
    grade[(mqs >= 70) & (mqs < 80)] = 2  # B
    grade[(mqs >= 55) & (mqs < 70)] = 3  # C
    grade[mqs < 55] = 4                  # D
    return grade
