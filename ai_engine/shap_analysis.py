# -*- coding: utf-8 -*-
"""
shap_analysis.py — SHAP-based XAI for MDAF multimodal model.
Modality-level and feature-level mean |SHAP| contributions.
"""

import torch, numpy as np
import shap
from .dataset import MSCDataset


MODALITY_FEATURE_MAP = {
    'M1_morph': MSCDataset.M1_MORPH_COLS,
    'M2':       MSCDataset.M2_COLS,
    'M3':       MSCDataset.M3_COLS,
    'M4':       MSCDataset.M4_COLS,
    'M5':       MSCDataset.M5_COLS,
}

ALL_FEATURE_COLS = (MSCDataset.M1_MORPH_COLS + MSCDataset.M2_COLS +
                    MSCDataset.M3_COLS + MSCDataset.M4_COLS + MSCDataset.M5_COLS)


class MQSWrapper:
    """Wraps MDAF model for SHAP: tabular input (35-d) → MQS scalar."""
    def __init__(self, model, device, cnn_embed_mean):
        self.model  = model.eval()
        self.device = device
        self.cnn_mean = torch.tensor(cnn_embed_mean, dtype=torch.float32).unsqueeze(0)

        n1 = len(MSCDataset.M1_MORPH_COLS)
        n2 = n1 + len(MSCDataset.M2_COLS)
        n3 = n2 + len(MSCDataset.M3_COLS)
        n4 = n3 + len(MSCDataset.M4_COLS)
        self.slices = {
            'm1_morph': (0, n1), 'm2': (n1, n2),
            'm3': (n2, n3), 'm4': (n3, n4), 'm5': (n4, None),
        }

    def __call__(self, X):
        """X: np.ndarray (n, 35)"""
        B = X.shape[0]
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        cnn = self.cnn_mean.to(self.device).expand(B, -1)

        batch = {
            'm1_morph': X_t[:, self.slices['m1_morph'][0]:self.slices['m1_morph'][1]],
            'm1_cnn'  : cnn,
            'm2'      : X_t[:, self.slices['m2'][0]:self.slices['m2'][1]],
            'm3'      : X_t[:, self.slices['m3'][0]:self.slices['m3'][1]],
            'm4'      : X_t[:, self.slices['m4'][0]:self.slices['m4'][1]],
            'm5'      : X_t[:, self.slices['m5'][0]:],
        }
        with torch.no_grad():
            mqs, _, _, _, _ = self.model(batch)
        return mqs.cpu().numpy()


def run_shap_analysis(model, ds_test, device, n_background=100, n_explain=200):
    """
    Compute SHAP values for tabular features (35-d, excluding CNN embedding).
    Returns:
        shap_vals: np.ndarray (n_explain, 35)
        modality_contrib: dict {'M1_morph': mean_abs_shap, ...}
        feature_contrib: dict {feature_name: mean_abs_shap}
    """
    print('\n[SHAP] Computing SHAP values...')

    # Build tabular arrays
    def get_X(ds):
        rows = [np.concatenate([
            ds.X_m1_morph[i], ds.X_m2[i],
            ds.X_m3[i], ds.X_m4[i], ds.X_m5[i]
        ]) for i in range(len(ds))]
        return np.stack(rows)

    X_all = get_X(ds_test)
    cnn_mean = ds_test.cnn_embed.mean(axis=0)

    wrapper = MQSWrapper(model, device, cnn_mean)

    # Background dataset (KernelExplainer)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_all), size=min(n_background, len(X_all)), replace=False)
    X_bg   = X_all[bg_idx]
    ex_idx = rng.choice(len(X_all), size=min(n_explain, len(X_all)), replace=False)
    X_ex   = X_all[ex_idx]

    explainer  = shap.KernelExplainer(wrapper, X_bg)
    shap_vals  = explainer.shap_values(X_ex, nsamples=100, silent=True)

    # Modality-level contributions
    modality_contrib = {}
    offset = 0
    for mod_name, cols in MODALITY_FEATURE_MAP.items():
        n = len(cols)
        contrib = np.abs(shap_vals[:, offset:offset+n]).mean()
        modality_contrib[mod_name] = round(float(contrib), 4)
        offset += n

    # Normalize to %
    total = sum(modality_contrib.values())
    modality_pct = {k: round(v/total*100, 1) for k, v in modality_contrib.items()}

    # Feature-level contributions
    feature_contrib = {}
    for i, feat in enumerate(ALL_FEATURE_COLS):
        feature_contrib[feat] = round(float(np.abs(shap_vals[:, i]).mean()), 4)

    print('[SHAP] Modality contributions (%):')
    for k, v in sorted(modality_pct.items(), key=lambda x: -x[1]):
        print(f'  {k:15s}: {v:5.1f}%')

    print('[SHAP] Top-10 features:')
    top10 = sorted(feature_contrib.items(), key=lambda x: -x[1])[:10]
    for feat, val in top10:
        print(f'  {feat:25s}: {val:.4f}')

    return shap_vals, modality_pct, feature_contrib
