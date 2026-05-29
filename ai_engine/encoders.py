# -*- coding: utf-8 -*-
"""
encoders.py
-----------
M1 (morphology + CNN), M2 (FLIM metabolic), M3 (donor health),
M4 (flow cytometry), M5 (manufacturing) modality encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphEncoder(nn.Module):
    """
    M1: Cellpose morphological scalars (6-d) + ResNet CNN embedding (512-d)
    → Linear projection to 512-d shared space
    """
    def __init__(self, morph_dim=6, cnn_dim=512, out_dim=512, dropout=0.2):
        super().__init__()
        # Morphological scalars branch
        self.morph_branch = nn.Sequential(
            nn.Linear(morph_dim, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(), nn.BatchNorm1d(128),
        )
        # CNN embedding branch (ResNet-50 output: 512-d)
        self.cnn_branch = nn.Sequential(
            nn.Linear(cnn_dim, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 384),
            nn.ReLU(), nn.BatchNorm1d(384),
        )
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 384, out_dim),
            nn.ReLU(), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x_morph, x_cnn):
        h_m = self.morph_branch(x_morph)
        h_c = self.cnn_branch(x_cnn)
        return self.fusion(torch.cat([h_m, h_c], dim=-1))


class MetabolicEncoder(nn.Module):
    """
    M2: FLIM-derived (NAD+ proxy score, tau_mean, DT) → 128-d
    3-layer MLP: input→64→128→128
    """
    def __init__(self, in_dim=3, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.ReLU(), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DonorEncoder(nn.Module):
    """
    M3: 15-item donor blood panel → 256-d
    4-layer MLP
    """
    def __init__(self, in_dim=15, out_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, 192),
            nn.ReLU(), nn.BatchNorm1d(192), nn.Dropout(dropout),
            nn.Linear(192, out_dim),
            nn.ReLU(), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlowCytometryEncoder(nn.Module):
    """
    M4: Flow cytometry (CD90/CD73/CD105 pos, CD34/CD45 neg, viability) → 64-d
    2-layer MLP
    """
    def __init__(self, in_dim=6, out_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU(), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ManufacturingEncoder(nn.Module):
    """
    M5: Manufacturing metadata (passage, culture_days, freeze_cycles,
        transport_h, seeding_ok) → 64-d
    2-layer MLP
    """
    def __init__(self, in_dim=5, out_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU(), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityProjection(nn.Module):
    """Project each modality to shared 256-d space for Cross-Attention."""
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        return self.norm(F.gelu(self.proj(x)))
