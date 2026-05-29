# -*- coding: utf-8 -*-
"""
train.py — Training loop with validation, early stopping, checkpointing.
"""

import torch, os, time, json
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

from .model import MDAFModel, MDAFLoss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = mqs_loss = grade_loss = 0
    mqs_preds, mqs_trues, grade_preds, grade_trues = [], [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        mqs_pred, grade_logits, tau_pred, _, _ = model(batch)
        loss, l_mqs, l_grade = criterion(
            mqs_pred, grade_logits, tau_pred,
            batch['mqs'], batch['grade'],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        mqs_loss   += l_mqs.item()
        grade_loss += l_grade.item()
        mqs_preds.extend(mqs_pred.detach().cpu().numpy())
        mqs_trues.extend(batch['mqs'].cpu().numpy())
        grade_preds.extend(grade_logits.argmax(-1).detach().cpu().numpy())
        grade_trues.extend(batch['grade'].cpu().numpy())

    n = len(loader)
    mqs_arr = np.array(mqs_preds); mqs_t = np.array(mqs_trues)
    ss_res = ((mqs_arr - mqs_t)**2).sum()
    ss_tot = ((mqs_t - mqs_t.mean())**2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    f1 = f1_score(grade_trues, grade_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss/n, 'mqs_loss': mqs_loss/n,
        'grade_loss': grade_loss/n, 'r2': r2, 'f1': f1,
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    mqs_preds, mqs_trues, grade_preds, grade_trues = [], [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mqs_pred, grade_logits, tau_pred, _, _ = model(batch)
        loss, _, _ = criterion(mqs_pred, grade_logits, tau_pred,
                                batch['mqs'], batch['grade'])
        total_loss += loss.item()
        mqs_preds.extend(mqs_pred.cpu().numpy())
        mqs_trues.extend(batch['mqs'].cpu().numpy())
        grade_preds.extend(grade_logits.argmax(-1).cpu().numpy())
        grade_trues.extend(batch['grade'].cpu().numpy())

    n = len(loader)
    mqs_arr = np.array(mqs_preds); mqs_t = np.array(mqs_trues)
    ss_res = ((mqs_arr - mqs_t)**2).sum()
    ss_tot = ((mqs_t - mqs_t.mean())**2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    rmse = np.sqrt(((mqs_arr - mqs_t)**2).mean())
    pearson_r, _ = pearsonr(mqs_arr, mqs_t)
    f1 = f1_score(grade_trues, grade_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss/n, 'r2': r2, 'rmse': rmse,
        'pearson_r': pearson_r, 'f1': f1,
        'preds': mqs_arr, 'trues': mqs_t,
        'grade_preds': np.array(grade_preds),
        'grade_trues': np.array(grade_trues),
    }


def train(model, train_loader, val_loader, device,
          epochs=60, lr=1e-3, weight_decay=1e-4,
          patience=10, checkpoint_dir='checkpoints'):

    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = MDAFLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    wait = 0
    history = []

    print(f'\n[Train] Starting training: {epochs} epochs, lr={lr}')
    print(f'[Train] Device: {device}')

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'train': tr, 'val': va})

        if va['loss'] < best_val_loss:
            best_val_loss = va['loss']
            wait = 0
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pt')
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            dt = time.time() - t0
            print(f'  Epoch {epoch:3d}/{epochs} | '
                  f'tr_loss={tr["loss"]:.4f} tr_R²={tr["r2"]:.3f} tr_F1={tr["f1"]:.3f} | '
                  f'val_loss={va["loss"]:.4f} val_R²={va["r2"]:.3f} val_F1={va["f1"]:.3f} | '
                  f'{dt:.1f}s')

        if wait >= patience:
            print(f'[Train] Early stopping at epoch {epoch} (patience={patience})')
            break

    # Load best
    model.load_state_dict(torch.load(f'{checkpoint_dir}/best_model.pt',
                                      map_location=device))
    with open(f'{checkpoint_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f'[Train] Best val_loss={best_val_loss:.4f}')
    return model, history
