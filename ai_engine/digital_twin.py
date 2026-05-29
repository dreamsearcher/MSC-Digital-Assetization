# -*- coding: utf-8 -*-
"""
digital_twin.py — Patient-Batch Digital Twin Matching Module.
MatchScore = w1*Φ_inflam + w2*Φ_metab + w3*Φ_regen + w4*MQS_norm + penalty_safety
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


WEIGHTS = {'inflammation': 0.30, 'metabolic': 0.25,
           'regenerative': 0.25, 'mqs': 0.20}

TIER_MAP = [(90, 'Optimal',         '★★★'),
            (80, 'Suitable',        '★★☆'),
            (70, 'Conditional',     '★☆☆'),
            (0,  'Not Recommended', '☆☆☆')]


@dataclass
class PatientProfile:
    patient_id: str
    # Inflammation
    hsCRP: float       # mg/L
    IL6:   float       # pg/mL
    TNF_a: float       # pg/mL
    # Metabolic
    glucose:  float    # mg/dL
    HbA1c:    float    # %
    HOMA_IR:  float
    # Clinical demand
    disease_severity: float       # 0-1 (higher = more severe)
    tissue_damage_score: float    # 0-1
    treatment_type: str           # 'immune' | 'regenerative' | 'metabolic'
    # Safety constraints
    age: int
    bmi: float
    contraindications: list = field(default_factory=list)


@dataclass
class BatchProfile:
    batch_id:    str
    mqs:         float       # 0-100
    grade:       str         # S/A/B/C/D
    passage:     int
    viability:   float       # %
    cd90_fold:   float
    cd73_fold:   float
    DT:          float       # h/passage
    nadp_proxy:  float       # normalized score
    tau_mean:    float       # ns
    z_embedding: Optional[np.ndarray] = None   # 256-d from model


def _inflammation_phi(patient: PatientProfile, batch: BatchProfile) -> float:
    """How well does this batch address patient's inflammation profile?"""
    # High CRP/IL6 patients need stronger immunosuppression → prefer S/A grade
    inflam_burden = np.clip((patient.hsCRP/15 + patient.IL6/20 + patient.TNF_a/15) / 3, 0, 1)
    batch_immuno  = np.clip(batch.mqs / 100, 0, 1)
    return float(batch_immuno * (0.6 + 0.4*inflam_burden))


def _metabolic_phi(patient: PatientProfile, batch: BatchProfile) -> float:
    """Metabolic compatibility."""
    glucose_risk = np.clip((patient.glucose - 70) / 130, 0, 1)
    hba1c_risk   = np.clip((patient.HbA1c - 4.5) / 6, 0, 1)
    metab_risk   = (glucose_risk + hba1c_risk) / 2
    # High metabolic risk patients benefit more from high-NAD+ proxy batches
    nadp_benefit = np.clip(batch.nadp_proxy, 0, 1)  # already normalized
    dt_ok = 1 - np.clip((batch.DT - 12) / 60, 0, 1)  # lower DT = better proliferation
    return float((nadp_benefit * 0.5 + dt_ok * 0.3 + (1-metab_risk) * 0.2))


def _regenerative_phi(patient: PatientProfile, batch: BatchProfile) -> float:
    """Regenerative match based on disease severity and batch potency."""
    sev   = patient.disease_severity
    dmg   = patient.tissue_damage_score
    demand = (sev + dmg) / 2
    # Severe patients need highest-grade batches
    grade_val = {'S': 1.0, 'A': 0.85, 'B': 0.70, 'C': 0.50, 'D': 0.20}.get(batch.grade, 0.5)
    # Treatment type alignment
    type_bonus = 0.1 if (
        (patient.treatment_type == 'immune'       and batch.cd73_fold > 15) or
        (patient.treatment_type == 'regenerative' and batch.cd90_fold > 120) or
        (patient.treatment_type == 'metabolic'    and batch.nadp_proxy > 0.6)
    ) else 0.0
    return float(min(1.0, grade_val * (0.7 + 0.3*demand) + type_bonus))


def _safety_penalty(patient: PatientProfile, batch: BatchProfile) -> float:
    """Safety penalty (0 = no penalty, reduces match score)."""
    penalty = 0.0
    if batch.passage > 5:
        penalty += 0.15
    if batch.viability < 85:
        penalty += 0.10 * ((85 - batch.viability) / 15)
    if patient.age > 70 and batch.grade in ('C', 'D'):
        penalty += 0.20
    if 'immunosuppressant' in patient.contraindications and batch.grade == 'D':
        penalty += 0.30
    return min(penalty, 0.40)


def compute_match_score(patient: PatientProfile, batch: BatchProfile,
                        weights: dict = WEIGHTS) -> dict:
    """
    Compute MatchScore ∈ [0, 100].
    Returns full breakdown for transparency.
    """
    phi_inf  = _inflammation_phi(patient, batch)
    phi_met  = _metabolic_phi(patient, batch)
    phi_reg  = _regenerative_phi(patient, batch)
    mqs_norm = batch.mqs / 100.0
    penalty  = _safety_penalty(patient, batch)

    raw_score = (
        weights['inflammation'] * phi_inf  +
        weights['metabolic']    * phi_met  +
        weights['regenerative'] * phi_reg  +
        weights['mqs']          * mqs_norm
    )
    score = max(0.0, raw_score - penalty) * 100.0

    # Tier
    tier = 'Not Recommended'
    for threshold, t_name, _ in TIER_MAP:
        if score >= threshold:
            tier = t_name
            break

    return {
        'patient_id':    patient.patient_id,
        'batch_id':      batch.batch_id,
        'match_score':   round(score, 2),
        'tier':          tier,
        'phi_inflammation': round(phi_inf*100, 2),
        'phi_metabolic':    round(phi_met*100, 2),
        'phi_regenerative': round(phi_reg*100, 2),
        'mqs_component':    round(mqs_norm*100, 2),
        'safety_penalty':   round(penalty*100, 2),
        'batch_grade':      batch.grade,
        'batch_mqs':        batch.mqs,
    }


def rank_batches_for_patient(patient: PatientProfile,
                              batches: list[BatchProfile],
                              top_k: int = 5) -> list[dict]:
    """Rank all available batches for a given patient, return top-k."""
    scores = [compute_match_score(patient, b) for b in batches]
    ranked = sorted(scores, key=lambda x: -x['match_score'])[:top_k]
    print(f'\n[DigitalTwin] Top-{top_k} batches for patient {patient.patient_id}:')
    for i, r in enumerate(ranked, 1):
        print(f'  #{i:2d} batch={r["batch_id"]}  score={r["match_score"]:5.1f}  '
              f'tier={r["tier"]:18s}  grade={r["batch_grade"]}  MQS={r["batch_mqs"]:.1f}')
    return ranked


def generate_demo_patients_and_batches(df, n_patients=4):
    """Generate demo patient profiles and batch profiles from synthetic data."""
    rng = np.random.default_rng(42)

    patients = [
        PatientProfile('P001', hsCRP=12.0, IL6=18.0, TNF_a=12.0,
                        glucose=180, HbA1c=8.5, HOMA_IR=4.5,
                        disease_severity=0.9, tissue_damage_score=0.8,
                        treatment_type='immune', age=55, bmi=26.0,
                        contraindications=[]),
        PatientProfile('P002', hsCRP=8.0, IL6=12.0, TNF_a=8.0,
                        glucose=160, HbA1c=7.8, HOMA_IR=3.8,
                        disease_severity=0.85, tissue_damage_score=0.75,
                        treatment_type='metabolic', age=62, bmi=30.0,
                        contraindications=[]),
        PatientProfile('P003', hsCRP=6.0, IL6=8.0, TNF_a=5.0,
                        glucose=130, HbA1c=6.5, HOMA_IR=2.5,
                        disease_severity=0.75, tissue_damage_score=0.80,
                        treatment_type='regenerative', age=48, bmi=23.0,
                        contraindications=[]),
        PatientProfile('P004', hsCRP=2.5, IL6=3.0, TNF_a=2.0,
                        glucose=95, HbA1c=5.2, HOMA_IR=1.2,
                        disease_severity=0.40, tissue_damage_score=0.35,
                        treatment_type='regenerative', age=38, bmi=22.0,
                        contraindications=[]),
    ]

    # Build batch profiles from top-grade synthetic batches
    top_df = df.nlargest(20, 'MQS')
    batches = []
    for _, row in top_df.iterrows():
        from .dataset import GRADE_INV
        grade_str = row['grade_str'] if 'grade_str' in row else 'B'
        batches.append(BatchProfile(
            batch_id   = f"B{int(row.name):04d}",
            mqs        = float(row['MQS']),
            grade      = grade_str,
            passage    = int(row['passage']),
            viability  = float(row['viability']),
            cd90_fold  = float(row['cd90_fold']),
            cd73_fold  = float(row['cd73_fold']),
            DT         = float(row['DT']),
            nadp_proxy = float(np.clip(row['nadp_proxy'] / 600, 0, 1)),
            tau_mean   = float(row['tau_mean']),
        ))

    return patients, batches
