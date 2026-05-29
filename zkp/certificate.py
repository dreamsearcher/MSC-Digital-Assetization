# -*- coding: utf-8 -*-
"""
certificate.py — MSC Digital Certificate builder.
On-chain fields + off-chain data structure.
"""

import hashlib, json, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from zkp.zkp_simulator import Groth16Proof, sha256_hash


CERTIFICATE_STATUS = {0: 'Valid', 1: 'Superseded', 2: 'Revoked'}

LIFECYCLE_EVENTS = [
    'DonorConsentRegistered', 'TissueCollected', 'MSCIsolated',
    'CultureInitiated', 'PassageUpdated', 'QualityAssessmentCompleted',
    'MQSComputed', 'CertificateIssued', 'ZKPVerificationCompleted',
    'StorageEventRecorded', 'TransportEventRecorded',
    'ClinicalReleaseRequested', 'DigitalTwinMatchRecorded',
    'ClinicalUseCompleted', 'CertificateRevoked',
]


@dataclass
class OnChainCertificate:
    """Fields stored on-chain (immutable, public)."""
    cert_id:            str
    batch_hash:         str
    mqs_grade_hash:     str      # hash(MQS || grade)
    data_hash:          str      # SHA-256(off-chain data package)
    issuer_did:         str
    release_zkp_proof:  Optional[str]   # REP π bytes (hex)
    premium_zkp_proof:  Optional[str]   # PQP π bytes (hex), None if Grade B
    zkp_policy_id:      str             # '0x01' REP / '0x02' REP+PQP
    lifecycle_event_hash: str
    passage_number:     int
    cell_source_code:   str      # 'BM' | 'AT' | 'UCB'
    model_version:      str
    issue_timestamp:    float
    certificate_status: int      # 0=Valid, 1=Superseded, 2=Revoked
    mqs_grade:          str      # grade letter (for QualityGateway pre-filter)


@dataclass
class OffChainDataPackage:
    """Fields stored off-chain (AES-256 encrypted, hash anchored on-chain)."""
    batch_id:          str
    mqs_value:         float
    grade:             str
    viability:         float
    tau_mean:          float
    nadp_proxy_score:  float
    DT:                float
    passage:           int
    morphology_report: dict
    flow_cytometry:    dict
    flim_report:       dict
    donor_panel:       dict
    culture_history:   list
    shap_report:       dict
    gmp_records:       dict
    created_at:        float


def _sha256(data) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _did(institution_id: str) -> str:
    return f'did:mdaf:{institution_id}:{hashlib.sha256(institution_id.encode()).hexdigest()[:16]}'


def build_certificate(
    batch_row,           # pandas Series from synthetic data
    mqs: float,
    grade: str,
    proof_rep: Optional[Groth16Proof],
    proof_pqp: Optional[Groth16Proof],
    institution_id: str = 'biobank_001',
    cell_source: str = 'UCB',
    model_version: str = 'MDAFModel-v1.0',
) -> tuple:
    """
    Build OnChainCertificate + OffChainDataPackage.
    Returns (on_chain, off_chain, cert_id)
    """
    cert_id   = 'CERT-' + uuid.uuid4().hex[:16].upper()
    batch_id  = f'BATCH-{int(batch_row.name):06d}'
    timestamp = time.time()

    # Off-chain package
    off_chain = OffChainDataPackage(
        batch_id        = batch_id,
        mqs_value       = float(mqs),
        grade           = grade,
        viability       = float(batch_row['viability']),
        tau_mean        = float(batch_row['tau_mean']),
        nadp_proxy_score= float(batch_row['nadp_proxy']),
        DT              = float(batch_row['DT']),
        passage         = int(batch_row['passage']),
        morphology_report = {
            'cell_area':         float(batch_row['cell_area']),
            'aspect_ratio':      float(batch_row['aspect_ratio']),
            'cn_ratio':          float(batch_row['cn_ratio']),
            'filop_length':      float(batch_row['filop_length']),
            'nuc_circularity':   float(batch_row['nuc_circularity']),
            'boundary_sharpness':float(batch_row['boundary_sharpness']),
        },
        flow_cytometry = {
            'cd90_fold': float(batch_row['cd90_fold']),
            'cd73_fold': float(batch_row['cd73_fold']),
            'cd105_fold':float(batch_row['cd105_fold']),
            'cd34_neg':  float(batch_row['cd34_score']),
            'cd45_neg':  float(batch_row['cd45_score']),
            'viability': float(batch_row['viability']),
        },
        flim_report = {
            'tau_mean_ns':      float(batch_row['tau_mean']),
            'nadp_proxy_score': float(batch_row['nadp_proxy']),
            'measurement_conditions': 'NADH 2P excitation 740nm',
        },
        donor_panel = {
            'hsCRP': float(batch_row['hsCRP']),
            'IL6':   float(batch_row['IL6']),
            'BMI':   float(batch_row['BMI']),
            'HbA1c': float(batch_row['HbA1c']),
        },
        culture_history = [
            {'event': 'CultureInitiated', 'passage': 1, 'timestamp': timestamp - 86400*14},
            {'event': 'PassageUpdated',   'passage': int(batch_row['passage']), 'timestamp': timestamp},
        ],
        shap_report  = {'status': 'generated_separately'},
        gmp_records  = {'compliant': True, 'audit_code': 'GMP-2025-001'},
        created_at   = timestamp,
    )

    off_chain_dict = asdict(off_chain)
    data_hash = _sha256(off_chain_dict)
    mqs_grade_hash = _sha256(f'{mqs:.4f}_{grade}')
    batch_hash     = _sha256(batch_id)
    lifecycle_hash = _sha256({'event': 'CertificateIssued', 'cert_id': cert_id,
                               'timestamp': timestamp})

    # ZKP policy
    if proof_pqp is not None and grade in ('S', 'A'):
        policy_id   = '0x03'   # REP + PQP
        rep_bytes   = proof_rep.proof_bytes if proof_rep else None
        pqp_bytes   = proof_pqp.proof_bytes
    elif proof_rep is not None:
        policy_id   = '0x01'   # REP only
        rep_bytes   = proof_rep.proof_bytes
        pqp_bytes   = None
    else:
        policy_id   = '0x00'
        rep_bytes   = None
        pqp_bytes   = None

    on_chain = OnChainCertificate(
        cert_id            = cert_id,
        batch_hash         = batch_hash,
        mqs_grade_hash     = mqs_grade_hash,
        data_hash          = data_hash,
        issuer_did         = _did(institution_id),
        release_zkp_proof  = rep_bytes,
        premium_zkp_proof  = pqp_bytes,
        zkp_policy_id      = policy_id,
        lifecycle_event_hash = lifecycle_hash,
        passage_number     = int(batch_row['passage']),
        cell_source_code   = cell_source,
        model_version      = model_version,
        issue_timestamp    = timestamp,
        certificate_status = 0,
        mqs_grade          = grade,
    )

    return on_chain, off_chain, cert_id, data_hash
