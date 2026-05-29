# -*- coding: utf-8 -*-
"""
dataset.py
----------
Synthetic MSC batch data generation (n=2000, seed=42, rule-engine based)
and PyTorch Dataset class for M1-M5 modality inputs.

Rule engine design:
  MQS_base = 0.27*morph + 0.31*metabolic + 0.19*flow + 0.13*manuf + 0.10*donor
  MQS = clip(MQS_base + N(0,5), 0, 100)
  Grade S:90-100 / A:80-89 / B:70-79 / C:55-69 / D:0-54
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os, pickle


# ── Constants ────────────────────────────────────────────────────────────────
GRADE_MAP   = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
GRADE_INV   = {v: k for k, v in GRADE_MAP.items()}
N_SAMPLES   = 2000
SEED        = 42
NOISE_STD   = 4.0

# Modality dimensions
DIM_M1_MORPH = 6      # morphological scalars from Cellpose
DIM_M1_CNN   = 512    # ResNet-50 embedding (simulated)
DIM_M2       = 3      # NAD+ proxy score, tau_mean, DT
DIM_M3       = 15     # donor blood panel + BMI
DIM_M4       = 6      # flow cytometry (CD73/CD90/CD105 pos, CD34/CD45 neg, viability)
DIM_M5       = 5      # manufacturing metadata


# ── Rule Engine ───────────────────────────────────────────────────────────────
def _morph_score(df):
    """High morphology score = large area, good circularity, long filopodia, sharp edges"""
    area_n    = np.clip((df['cell_area'] - 200) / 1800, 0, 1)
    ar_n      = 1 - np.clip((df['aspect_ratio'] - 1) / 9, 0, 1)   # closer to 1 = round
    cn_n      = np.clip(df['cn_ratio'] / 3, 0, 1)
    fil_n     = np.clip(df['filop_length'] / 200, 0, 1)
    circ_n    = df['nuc_circularity']
    sharp_n   = df['boundary_sharpness']
    return (0.25*area_n + 0.15*ar_n + 0.20*cn_n +
            0.15*fil_n + 0.15*circ_n + 0.10*sharp_n) * 100

def _metabolic_score(df):
    """High metabolic score = high NAD+ proxy, low tau_mean, low DT"""
    nadp_n  = np.clip(df['nadp_proxy'] / 600, 0, 1)
    tau_n   = 1 - np.clip((df['tau_mean'] - 1.5) / 2.5, 0, 1)
    dt_n    = 1 - np.clip((df['DT'] - 12) / 60, 0, 1)
    return (0.40*nadp_n + 0.35*tau_n + 0.25*dt_n) * 100

def _flow_score(df):
    """High flow score = high CD90/CD73 positivity, low CD34/CD45, high viability"""
    cd90_n  = np.clip(df['cd90_fold'] / 200, 0, 1)
    cd73_n  = np.clip(df['cd73_fold'] / 25,  0, 1)
    cd105_n = np.clip(df['cd105_fold'] / 5,  0, 1)
    neg_n   = 1 - np.clip((df['cd34_score'] + df['cd45_score']) / 2, 0, 1)
    via_n   = np.clip((df['viability'] - 70) / 30, 0, 1)
    return (0.30*cd90_n + 0.25*cd73_n + 0.10*cd105_n +
            0.20*neg_n  + 0.15*via_n) * 100

def _manuf_score(df):
    """High manufacturing score = low passage, short transport, good freeze history"""
    pass_n  = 1 - np.clip((df['passage'] - 1) / 7, 0, 1)
    trans_n = 1 - np.clip(df['transport_h'] / 72, 0, 1)
    freeze_n= np.clip(1 - df['freeze_cycles'] / 3, 0, 1)
    dur_n   = np.clip((df['culture_days'] - 3) / 25, 0, 1)
    seed_n  = df['seeding_ok'].astype(float)
    return (0.35*pass_n + 0.25*trans_n + 0.20*freeze_n +
            0.10*dur_n  + 0.10*seed_n) * 100

def _donor_score(df):
    """High donor score = low inflammation, good metabolism, normal BMI"""
    crp_n   = 1 - np.clip(df['hsCRP'] / 15,     0, 1)
    il6_n   = 1 - np.clip(df['IL6'] / 20,        0, 1)
    glc_n   = 1 - np.clip((df['glucose'] - 70) / 130, 0, 1)
    hba1_n  = 1 - np.clip((df['HbA1c'] - 4.5) / 6, 0, 1)
    bmi_n   = 1 - np.clip(np.abs(df['BMI'] - 22) / 20, 0, 1)
    nk_n    = np.clip(df['NK_activity'] / 60, 0, 1)
    return (0.20*crp_n + 0.15*il6_n + 0.25*glc_n +
            0.15*hba1_n + 0.15*bmi_n + 0.10*nk_n) * 100


def mqs_to_grade(mqs):
    if   mqs >= 90: return 'S'
    elif mqs >= 80: return 'A'
    elif mqs >= 70: return 'B'
    elif mqs >= 55: return 'C'
    else:           return 'D'


# ── Data Generation ───────────────────────────────────────────────────────────
def generate_synthetic_data(n=N_SAMPLES, seed=SEED, save_path=None):
    rng = np.random.default_rng(seed)

    # ── M1: Morphological features ───────────────────────────
    passage_raw = rng.integers(1, 9, n)
    age_factor  = 1 + (passage_raw - 1) * 0.08

    cell_area        = rng.uniform(200, 2000, n) / age_factor
    aspect_ratio     = rng.uniform(1.0, 8.0, n) * age_factor * 0.5
    cn_ratio         = rng.uniform(0.5, 3.0, n) / age_factor
    filop_length     = rng.uniform(5, 200, n) / age_factor
    nuc_circularity  = rng.uniform(0.4, 1.0, n) / age_factor
    boundary_sharp   = rng.uniform(0.3, 1.0, n) / age_factor

    # ResNet-50 CNN embedding (512-d): simulated as Gaussian with morphology signal
    cnn_embed = rng.normal(0, 1, (n, DIM_M1_CNN)).astype(np.float32)
    morph_signal = np.stack([cell_area/2000, 1/aspect_ratio,
                              cn_ratio/3, filop_length/200,
                              nuc_circularity, boundary_sharp], axis=1)
    cnn_embed[:, :6] += morph_signal * 2.0   # inject morphology into first 6 dims

    # ── M2: FLIM metabolic-proliferative ─────────────────────
    nadp_proxy = rng.uniform(80, 600, n) / age_factor
    tau_mean   = rng.uniform(1.5, 4.0, n) * (1 + (age_factor - 1) * 0.3)
    DT         = rng.uniform(12, 72, n) * age_factor

    # ── M3: Donor blood panel ─────────────────────────────────
    hsCRP    = rng.uniform(0.1, 15, n)
    IL6      = rng.uniform(0.5, 20, n)
    TNF_a    = rng.uniform(0.5, 15, n)
    ESR      = rng.uniform(2, 40, n)
    glucose  = rng.uniform(70, 200, n)
    HbA1c    = rng.uniform(4.5, 10.5, n)
    HOMA_IR  = rng.uniform(0.5, 6.0, n)
    TC       = rng.uniform(120, 280, n)
    LDL      = rng.uniform(60, 180, n)
    HDL      = rng.uniform(30, 90, n)
    TG       = rng.uniform(50, 300, n)
    NK_act   = rng.uniform(5, 60, n)
    lymph_r  = rng.uniform(15, 45, n)
    IgG      = rng.uniform(600, 1800, n)
    BMI      = rng.uniform(18, 42, n)

    # ── M4: Flow cytometry ────────────────────────────────────
    cd90_fold  = rng.uniform(50, 200, n) / age_factor
    cd73_fold  = rng.uniform(5, 25, n) / age_factor
    cd105_fold = rng.uniform(1.0, 5.0, n)
    cd34_score = rng.uniform(0, 0.3, n)
    cd45_score = rng.uniform(0, 0.3, n)
    # Viability: base 70-99, slightly reduced by age but not below 50
    viability  = rng.uniform(70, 99, n) - (age_factor - 1) * 5
    viability  = np.clip(viability, 50, 99)

    # ── M5: Manufacturing metadata ────────────────────────────
    passage       = passage_raw.astype(float)
    culture_days  = rng.uniform(3, 28, n)
    freeze_cycles = rng.integers(0, 4, n).astype(float)
    transport_h   = rng.uniform(0, 72, n)
    seeding_ok    = (rng.uniform(0, 1, n) > 0.1).astype(float)

    # ── Assemble DataFrame ────────────────────────────────────
    df = pd.DataFrame({
        'cell_area': cell_area, 'aspect_ratio': aspect_ratio,
        'cn_ratio': cn_ratio, 'filop_length': filop_length,
        'nuc_circularity': nuc_circularity, 'boundary_sharpness': boundary_sharp,
        'nadp_proxy': nadp_proxy, 'tau_mean': tau_mean, 'DT': DT,
        'hsCRP': hsCRP, 'IL6': IL6, 'TNF_a': TNF_a, 'ESR': ESR,
        'glucose': glucose, 'HbA1c': HbA1c, 'HOMA_IR': HOMA_IR,
        'TC': TC, 'LDL': LDL, 'HDL': HDL, 'TG': TG,
        'NK_activity': NK_act, 'lymphocyte_ratio': lymph_r,
        'IgG': IgG, 'BMI': BMI,
        'cd90_fold': cd90_fold, 'cd73_fold': cd73_fold,
        'cd105_fold': cd105_fold, 'cd34_score': cd34_score,
        'cd45_score': cd45_score, 'viability': viability,
        'passage': passage, 'culture_days': culture_days,
        'freeze_cycles': freeze_cycles, 'transport_h': transport_h,
        'seeding_ok': seeding_ok,
    })

    # ── Rule engine MQS ───────────────────────────────────────
    ms = _morph_score(df)
    me = _metabolic_score(df)
    fl = _flow_score(df)
    mn = _manuf_score(df)
    do = _donor_score(df)

    # Add offset to push distribution to realistic range
    mqs_base = (0.27*ms + 0.31*me + 0.19*fl + 0.13*mn + 0.10*do) + 20
    noise    = rng.normal(0, NOISE_STD, n)
    mqs      = np.clip(mqs_base + noise, 0, 100)

    df['MQS']       = mqs.astype(np.float32)
    df['grade_str'] = [mqs_to_grade(v) for v in mqs]
    df['grade']     = [GRADE_MAP[g] for g in df['grade_str']]

    # ── CNN embeddings (separate array) ───────────────────────
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(f'{save_path}/synthetic_msc.csv', index=False)
        np.save(f'{save_path}/cnn_embeddings.npy', cnn_embed)
        print(f'[Dataset] Saved {n} samples → {save_path}/')

    # Grade distribution
    grade_counts = df['grade_str'].value_counts().to_dict()
    print(f'[Dataset] n={n} | MQS mean={mqs.mean():.1f} std={mqs.std():.1f}')
    print(f'[Dataset] Grade dist: {grade_counts}')

    return df, cnn_embed


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class MSCDataset(Dataset):
    """
    Returns dict of tensors for each modality:
      m1_morph  : (6,)     morphological scalars
      m1_cnn    : (512,)   ResNet-50 CNN embedding
      m2        : (3,)     FLIM metabolic-proliferative
      m3        : (15,)    donor blood panel
      m4        : (6,)     flow cytometry
      m5        : (5,)     manufacturing metadata
      mqs       : scalar   regression target [0,100]
      grade     : long     classification target [0-4]
    """
    M1_MORPH_COLS = ['cell_area','aspect_ratio','cn_ratio',
                     'filop_length','nuc_circularity','boundary_sharpness']
    M2_COLS       = ['nadp_proxy','tau_mean','DT']
    M3_COLS       = ['hsCRP','IL6','TNF_a','ESR','glucose','HbA1c','HOMA_IR',
                     'TC','LDL','HDL','TG','NK_activity','lymphocyte_ratio','IgG','BMI']
    M4_COLS       = ['cd90_fold','cd73_fold','cd105_fold',
                     'cd34_score','cd45_score','viability']
    M5_COLS       = ['passage','culture_days','freeze_cycles','transport_h','seeding_ok']

    def __init__(self, df, cnn_embed, scalers=None, fit_scalers=True):
        self.df        = df.reset_index(drop=True)
        self.cnn_embed = cnn_embed.astype(np.float32)

        all_cols = (self.M1_MORPH_COLS + self.M2_COLS +
                    self.M3_COLS + self.M4_COLS + self.M5_COLS)
        X = df[all_cols].values.astype(np.float32)

        if fit_scalers:
            self.scalers = {}
            for name, cols in [('m1', self.M1_MORPH_COLS), ('m2', self.M2_COLS),
                                ('m3', self.M3_COLS), ('m4', self.M4_COLS),
                                ('m5', self.M5_COLS)]:
                sc = StandardScaler()
                start = all_cols.index(cols[0])
                end   = start + len(cols)
                X[:, start:end] = sc.fit_transform(X[:, start:end])
                self.scalers[name] = sc
        else:
            assert scalers is not None
            self.scalers = scalers
            for name, cols in [('m1', self.M1_MORPH_COLS), ('m2', self.M2_COLS),
                                ('m3', self.M3_COLS), ('m4', self.M4_COLS),
                                ('m5', self.M5_COLS)]:
                start = all_cols.index(cols[0])
                end   = start + len(cols)
                X[:, start:end] = scalers[name].transform(X[:, start:end])

        n1m = len(self.M1_MORPH_COLS)
        n2  = n1m + len(self.M2_COLS)
        n3  = n2  + len(self.M3_COLS)
        n4  = n3  + len(self.M4_COLS)

        self.X_m1_morph = X[:, :n1m]
        self.X_m2       = X[:, n1m:n2]
        self.X_m3       = X[:, n2:n3]
        self.X_m4       = X[:, n3:n4]
        self.X_m5       = X[:, n4:]
        self.y_mqs      = df['MQS'].values.astype(np.float32)
        self.y_grade    = df['grade'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'm1_morph': torch.tensor(self.X_m1_morph[idx]),
            'm1_cnn'  : torch.tensor(self.cnn_embed[idx]),
            'm2'      : torch.tensor(self.X_m2[idx]),
            'm3'      : torch.tensor(self.X_m3[idx]),
            'm4'      : torch.tensor(self.X_m4[idx]),
            'm5'      : torch.tensor(self.X_m5[idx]),
            'mqs'     : torch.tensor(self.y_mqs[idx]),
            'grade'   : torch.tensor(self.y_grade[idx]),
        }


def build_dataloaders(data_dir='data', batch_size=64):
    """Build train/val/test DataLoaders with 70/15/15 stratified split."""
    csv_path = f'{data_dir}/synthetic_msc.csv'
    emb_path = f'{data_dir}/cnn_embeddings.npy'

    if not os.path.exists(csv_path):
        df, cnn = generate_synthetic_data(save_path=data_dir)
    else:
        df  = pd.read_csv(csv_path)
        cnn = np.load(emb_path)
        print(f'[Dataset] Loaded {len(df)} samples from {data_dir}/')

    # Stratified split by grade
    idx = np.arange(len(df))
    idx_tv, idx_test = train_test_split(idx, test_size=0.15,
                                        stratify=df['grade'].values, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.15/0.85,
                                          stratify=df['grade'].iloc[idx_tv].values,
                                          random_state=SEED)

    ds_train = MSCDataset(df.iloc[idx_train], cnn[idx_train], fit_scalers=True)
    scalers  = ds_train.scalers
    ds_val   = MSCDataset(df.iloc[idx_val],  cnn[idx_val],   scalers=scalers, fit_scalers=False)
    ds_test  = MSCDataset(df.iloc[idx_test], cnn[idx_test],  scalers=scalers, fit_scalers=False)

    print(f'[Dataset] Split → train:{len(ds_train)} val:{len(ds_val)} test:{len(ds_test)}')

    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0),
        scalers,
        df, cnn,
        {'train': idx_train, 'val': idx_val, 'test': idx_test},
    )


if __name__ == '__main__':
    df, cnn = generate_synthetic_data(save_path='data')
