# -*- coding: utf-8 -*-
"""
evaluate.py — Test set evaluation, baseline comparison, ablation study.
"""

import torch, numpy as np, pandas as pd
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                               confusion_matrix, classification_report)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr
import torch.nn as nn

from .model import MDAFModel, MDAFLoss
from .dataset import GRADE_MAP, GRADE_INV


# ── Full model test evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model, test_loader, device):
    model.eval()
    criterion = MDAFLoss()
    mqs_preds, mqs_trues, grade_preds, grade_probs, grade_trues = [], [], [], [], []
    embeddings = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mqs_pred, grade_logits, _, z_fusion, _ = model(batch)
        mqs_preds.extend(mqs_pred.cpu().numpy())
        mqs_trues.extend(batch['mqs'].cpu().numpy())
        grade_preds.extend(grade_logits.argmax(-1).cpu().numpy())
        grade_probs.extend(torch.softmax(grade_logits, -1).cpu().numpy())
        grade_trues.extend(batch['grade'].cpu().numpy())
        embeddings.append(z_fusion.cpu().numpy())

    mqs_p = np.array(mqs_preds); mqs_t = np.array(mqs_trues)
    gp    = np.array(grade_preds); gt = np.array(grade_trues)
    gprob = np.array(grade_probs)

    r2      = 1 - ((mqs_p - mqs_t)**2).sum() / ((mqs_t - mqs_t.mean())**2).sum()
    rmse    = np.sqrt(((mqs_p - mqs_t)**2).mean())
    pr, _   = pearsonr(mqs_p, mqs_t)
    f1_5    = f1_score(gt, gp, average='macro', zero_division=0)
    acc     = accuracy_score(gt, gp)

    # 4-class (exclude Grade D=4)
    mask4   = gt < 4
    f1_4    = f1_score(gt[mask4], gp[mask4], average='macro', zero_division=0)

    # AUROC (one-vs-rest)
    classes = sorted(np.unique(gt))
    try:
        y_bin   = label_binarize(gt, classes=list(range(5)))
        auroc   = roc_auc_score(y_bin, gprob, multi_class='ovr', average='macro', labels=classes)
    except Exception:
        auroc = 0.0

    metrics = {
        'R2': round(float(r2), 4),
        'RMSE': round(float(rmse), 3),
        'Pearson_r': round(float(pr), 4),
        'Macro_F1_5class': round(float(f1_5), 4),
        'Macro_F1_4class': round(float(f1_4), 4),
        'AUROC': round(float(auroc), 4),
        'Accuracy': round(float(acc), 4),
    }

    print('\n[Evaluate] ── Test Set Results ─────────────────────')
    for k, v in metrics.items():
        print(f'  {k:25s}: {v}')
    labels_present = sorted(set(gt.tolist()) | set(gp.tolist()))
    tnames = [['S','A','B','C','D'][i] for i in labels_present]
    print(f'\n{classification_report(gt, gp, labels=labels_present, target_names=tnames, zero_division=0)}')

    emb_matrix = np.vstack(embeddings)
    return metrics, emb_matrix, mqs_p, gt


# ── Baseline models ───────────────────────────────────────────────────────────
def run_baselines(df, split_idx):
    """Train sklearn baselines on tabular features (all M1_morph+M2+M3+M4+M5)."""
    from .dataset import MSCDataset
    import numpy as np

    feature_cols = (MSCDataset.M1_MORPH_COLS + MSCDataset.M2_COLS +
                    MSCDataset.M3_COLS + MSCDataset.M4_COLS + MSCDataset.M5_COLS)

    X = df[feature_cols].values.astype(np.float32)
    y_mqs   = df['MQS'].values
    y_grade = df['grade'].values

    X_tr = X[split_idx['train']]; y_mqs_tr = y_mqs[split_idx['train']]
    y_gr_tr = y_grade[split_idx['train']]
    X_te = X[split_idx['test']];  y_mqs_te = y_mqs[split_idx['test']]
    y_gr_te = y_grade[split_idx['test']]

    results = {}

    # Linear regression / Logistic regression
    lr  = LinearRegression().fit(X_tr, y_mqs_tr)
    lgr = LogisticRegression(max_iter=500, random_state=42).fit(X_tr, y_gr_tr)
    mqs_p = lr.predict(X_te)
    gr_p  = lgr.predict(X_te)
    r2  = 1 - ((mqs_p - y_mqs_te)**2).sum() / ((y_mqs_te - y_mqs_te.mean())**2).sum()
    f1  = f1_score(y_gr_te, gr_p, average='macro', zero_division=0)
    results['Linear/Logistic'] = {'R2': round(float(r2),3), 'Macro_F1': round(f1,3)}

    # XGBoost-style GradientBoosting
    gbr = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_tr, y_mqs_tr)
    gbc = GradientBoostingClassifier(n_estimators=200, random_state=42).fit(X_tr, y_gr_tr)
    mqs_p2 = gbr.predict(X_te)
    gr_p2  = gbc.predict(X_te)
    r2b = 1 - ((mqs_p2 - y_mqs_te)**2).sum() / ((y_mqs_te - y_mqs_te.mean())**2).sum()
    f1b = f1_score(y_gr_te, gr_p2, average='macro', zero_division=0)
    results['GradientBoosting (XGBoost-style)'] = {'R2': round(float(r2b),3), 'Macro_F1': round(f1b,3)}

    print('\n[Baselines] ── Baseline Comparison ─────────────────')
    for name, m in results.items():
        print(f'  {name:40s}: R²={m["R2"]:.3f}  F1={m["Macro_F1"]:.3f}')

    return results


# ── Ablation study ────────────────────────────────────────────────────────────
class AblatedModel(nn.Module):
    """Simplified MLP concatenation model for ablation (no Cross-Attention)."""
    def __init__(self, active_mods, proj_dim=256, dropout=0.2):
        super().__init__()
        self.active = active_mods
        dims = {
            'M1': 518,   # 6+512
            'M2': 3,
            'M3': 15,
            'M4': 6,
            'M5': 5,
        }
        in_dim = sum(dims[m] for m in active_mods)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 256),   nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, proj_dim), nn.ReLU(),
        )
        self.reg = nn.Sequential(nn.Linear(proj_dim,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
        self.cls = nn.Sequential(nn.Linear(proj_dim,64), nn.ReLU(), nn.Linear(64,5))

    def forward(self, batch):
        parts = []
        if 'M1' in self.active:
            parts += [batch['m1_morph'], batch['m1_cnn']]
        if 'M2' in self.active: parts.append(batch['m2'])
        if 'M3' in self.active: parts.append(batch['m3'])
        if 'M4' in self.active: parts.append(batch['m4'])
        if 'M5' in self.active: parts.append(batch['m5'])
        x = torch.cat(parts, dim=-1)
        z = self.net(x)
        return self.reg(z).squeeze(-1)*100, self.cls(z)


def run_ablation(train_loader, val_loader, test_loader, device, epochs=30):
    """Run ablation study over modality configurations."""
    configs = [
        (['M1'],                    'M1 only'),
        (['M2'],                    'M2 only'),
        (['M3'],                    'M3 only'),
        (['M1','M2'],               'M1+M2'),
        (['M1','M2','M3'],          'M1+M2+M3 (concat)'),
        (['M1','M2','M3','M4','M5'],'M1-M5 (concat, no attention)'),
    ]

    results = {}
    for mods, name in configs:
        model = AblatedModel(mods).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = MDAFLoss()

        for ep in range(epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                mqs_p, gl = model(batch)
                loss = 0.6*nn.MSELoss()(mqs_p, batch['mqs']) + \
                       0.4*nn.CrossEntropyLoss()(gl, batch['grade'])
                loss.backward()
                opt.step()

        # Evaluate
        model.eval()
        mqs_ps, mqs_ts, gps, gts = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mqs_p, gl = model(batch)
                mqs_ps.extend(mqs_p.cpu().numpy())
                mqs_ts.extend(batch['mqs'].cpu().numpy())
                gps.extend(gl.argmax(-1).cpu().numpy())
                gts.extend(batch['grade'].cpu().numpy())

        mp = np.array(mqs_ps); mt = np.array(mqs_ts)
        r2 = 1 - ((mp-mt)**2).sum() / ((mt-mt.mean())**2).sum()
        f1 = f1_score(gts, gps, average='macro', zero_division=0)
        results[name] = {'R2': round(float(r2),3), 'Macro_F1': round(f1,3)}
        print(f'  {name:45s}: R²={r2:.3f}  F1={f1:.3f}')

    return results
