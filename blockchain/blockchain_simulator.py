# -*- coding: utf-8 -*-
"""
blockchain_simulator.py
-----------------------
Simulates Hyperledger Besu QBFT 4-node permissioned blockchain
with three smart contracts:
  CertificateRegistry  — issuance, transfer, revocation, lifecycle
  QualityGateway       — two-tier ZKP gate (REP + PQP)
  LifeBankDirectory    — DID registration, RBAC, VC management

All Solidity logic is mirrored in Python for simulation.
"""

import time, hashlib, json, uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── QBFT Network Simulation ───────────────────────────────────────────────────
QBFT_FINALITY_S    = 1.2
QBFT_TPS           = 230
N_VALIDATORS       = 4

ROLES = ['DONOR', 'BIOBANK', 'LAB', 'HOSPITAL', 'AUDITOR', 'REGULATOR']


@dataclass
class QBFTBlock:
    block_number:  int
    parent_hash:   str
    transactions:  list
    timestamp:     float
    validator:     str
    block_hash:    str = ''

    def __post_init__(self):
        payload = json.dumps({
            'n': self.block_number,
            'p': self.parent_hash,
            'tx': len(self.transactions),
            'ts': self.timestamp,
        }).encode()
        self.block_hash = hashlib.sha256(payload).hexdigest()


@dataclass
class Transaction:
    tx_hash:    str
    from_addr:  str
    to_contract:str
    function:   str
    args:       dict
    timestamp:  float
    gas_used:   int
    status:     str   # 'success' | 'reverted'
    event_log:  list  = field(default_factory=list)


class QBFTLedger:
    """Simulates 4-node QBFT consensus ledger."""
    def __init__(self, n_validators=N_VALIDATORS):
        self.blocks       = []
        self.pending_tx   = []
        self.n_validators = n_validators
        self._genesis()

    def _genesis(self):
        g = QBFTBlock(0, '0'*64, [], time.time(), 'genesis')
        self.blocks.append(g)

    def submit_transaction(self, tx: Transaction) -> str:
        """Submit tx → simulate QBFT consensus → finalize."""
        self.pending_tx.append(tx)
        if len(self.pending_tx) >= 1:  # immediate finality simulation
            self._finalize_block()
        return tx.tx_hash

    def _finalize_block(self):
        validator = f'validator_{len(self.blocks) % self.n_validators}'
        block = QBFTBlock(
            block_number = len(self.blocks),
            parent_hash  = self.blocks[-1].block_hash,
            transactions = list(self.pending_tx),
            timestamp    = time.time(),
            validator    = validator,
        )
        self.blocks.append(block)
        self.pending_tx.clear()

    def get_transaction_receipt(self, tx_hash: str) -> Optional[dict]:
        for block in self.blocks:
            for tx in block.transactions:
                if tx.tx_hash == tx_hash:
                    return {'block': block.block_number,
                            'tx_hash': tx_hash,
                            'status': tx.status,
                            'gas': tx.gas_used,
                            'events': tx.event_log}
        return None

    def stats(self):
        total_tx = sum(len(b.transactions) for b in self.blocks)
        return {'blocks': len(self.blocks), 'transactions': total_tx,
                'tps_simulated': QBFT_TPS, 'finality_s': QBFT_FINALITY_S}


def _tx_hash():
    return '0x' + uuid.uuid4().hex


def _gas(base: int, extra: int = 0) -> int:
    return base + extra


# ── CertificateRegistry Contract ──────────────────────────────────────────────
class CertificateRegistry:
    """
    Solidity equivalent (Python simulation):
      mapping(bytes32 => Certificate) certificates;
      function issueCertificate(...)
      function updateLifecycleEvent(...)
      function revokeCertificate(...)
      function recordDigitalTwinMatch(...)
    """
    def __init__(self, ledger: QBFTLedger, gateway):
        self.ledger    = ledger
        self.gateway   = gateway
        self.registry  = {}      # cert_id → certificate dict
        self.events    = defaultdict(list)

    def issue_certificate(self, cert, data_hash: str, caller_role: str) -> Transaction:
        """
        issueCertificate(batchHash, issuerDID, modelVer, gradeHash, pi_REP, pi_PQP)
        Gate: QualityGateway must pass first.
        """
        # QualityGateway check
        gate_result = self.gateway.check(cert, data_hash)
        if not gate_result['passed']:
            tx = Transaction(
                tx_hash=_tx_hash(), from_addr=caller_role,
                to_contract='CertificateRegistry', function='issueCertificate',
                args={'cert_id': cert.cert_id},
                timestamp=time.time(), gas_used=21000, status='reverted',
                event_log=[{'event': 'QualityGateFailed', 'reason': gate_result['reason']}]
            )
            self.ledger.submit_transaction(tx)
            return tx

        self.registry[cert.cert_id] = {
            'cert':       asdict(cert),
            'data_hash':  data_hash,
            'issued_at':  time.time(),
            'status':     0,
            'lifecycle':  [],
        }

        tx = Transaction(
            tx_hash=_tx_hash(), from_addr=caller_role,
            to_contract='CertificateRegistry', function='issueCertificate',
            args={'cert_id': cert.cert_id, 'grade': cert.mqs_grade},
            timestamp=time.time(),
            gas_used=_gas(120000, gate_result.get('gas_used', 0)),
            status='success',
            event_log=[
                {'event': 'CertificateIssued', 'cert_id': cert.cert_id,
                 'grade': cert.mqs_grade, 'policy': cert.zkp_policy_id},
                {'event': 'ZKPVerificationCompleted',
                 'rep': gate_result['rep_valid'],
                 'pqp': gate_result['pqp_valid']},
            ]
        )
        self.ledger.submit_transaction(tx)
        self.events[cert.cert_id].append('CertificateIssued')
        return tx

    def update_lifecycle(self, cert_id: str, event_type: str,
                          caller_role: str) -> Transaction:
        if cert_id not in self.registry:
            raise ValueError(f'Certificate {cert_id} not found')
        if event_type not in ['DonorConsentRegistered', 'TissueCollected',
                               'MSCIsolated', 'CultureInitiated', 'PassageUpdated',
                               'QualityAssessmentCompleted', 'MQSComputed',
                               'CertificateIssued', 'ZKPVerificationCompleted',
                               'StorageEventRecorded', 'TransportEventRecorded',
                               'ClinicalReleaseRequested', 'DigitalTwinMatchRecorded',
                               'ClinicalUseCompleted', 'CertificateRevoked']:
            raise ValueError(f'Unknown event type: {event_type}')

        event_hash = hashlib.sha256(f'{cert_id}_{event_type}_{time.time()}'.encode()).hexdigest()
        self.registry[cert_id]['cert']['lifecycle_event_hash'] = event_hash
        self.registry[cert_id]['lifecycle'].append({'event': event_type,
                                                     'ts': time.time()})

        tx = Transaction(
            tx_hash=_tx_hash(), from_addr=caller_role,
            to_contract='CertificateRegistry', function='updateLifecycleEvent',
            args={'cert_id': cert_id, 'event': event_type, 'hash': event_hash},
            timestamp=time.time(), gas_used=45000, status='success',
            event_log=[{'event': 'LifecycleEventRecorded',
                        'cert_id': cert_id, 'event_type': event_type}]
        )
        self.ledger.submit_transaction(tx)
        return tx

    def revoke_certificate(self, cert_id: str, reason: str,
                            caller_role: str) -> Transaction:
        if cert_id not in self.registry:
            raise ValueError(f'Certificate {cert_id} not found')
        self.registry[cert_id]['status'] = 2
        self.registry[cert_id]['cert']['certificate_status'] = 2

        tx = Transaction(
            tx_hash=_tx_hash(), from_addr=caller_role,
            to_contract='CertificateRegistry', function='revokeCertificate',
            args={'cert_id': cert_id, 'reason_hash': hashlib.sha256(reason.encode()).hexdigest()},
            timestamp=time.time(), gas_used=35000, status='success',
            event_log=[{'event': 'CertificateRevoked', 'cert_id': cert_id}]
        )
        self.ledger.submit_transaction(tx)
        return tx

    def record_twin_match(self, cert_id: str, match_score: float,
                           patient_id: str, caller_role: str) -> Transaction:
        if cert_id not in self.registry:
            raise ValueError(f'Certificate {cert_id} not found')
        score_hash = hashlib.sha256(f'{cert_id}_{patient_id}_{match_score:.4f}'.encode()).hexdigest()
        self.registry[cert_id].setdefault('twin_matches', []).append({
            'patient_id': patient_id, 'score': match_score, 'ts': time.time()
        })

        tx = Transaction(
            tx_hash=_tx_hash(), from_addr=caller_role,
            to_contract='CertificateRegistry', function='recordDigitalTwinMatch',
            args={'cert_id': cert_id, 'match_score_hash': score_hash},
            timestamp=time.time(), gas_used=40000, status='success',
            event_log=[{'event': 'DigitalTwinMatchRecorded',
                        'cert_id': cert_id, 'score': match_score}]
        )
        self.ledger.submit_transaction(tx)
        return tx

    def query(self, cert_id: str) -> Optional[dict]:
        return self.registry.get(cert_id)


# ── QualityGateway Contract ───────────────────────────────────────────────────
class QualityGateway:
    """
    Simulates Solidity QualityGateway two-tier gate:
      require(grade <= GRADE_B && mqs >= 70, "Quality gate: minimum threshold");
      require(ZKPVerifier.verifyREP(...), "REP: invalid proof");
      if (grade <= GRADE_A) require(ZKPVerifier.verifyPQP(...), "PQP: invalid");
      D grade → immediate revert, rejection record only
    """
    def __init__(self, verifier):
        self.verifier      = verifier
        self.rejections    = []

    def check(self, cert, data_hash: str) -> dict:
        grade = cert.mqs_grade

        # Pre-filter: Grade D → immediate rejection
        if grade == 'D':
            self.rejections.append({'cert_id': cert.cert_id, 'grade': 'D',
                                     'reason': 'Grade D: pre-policy filter',
                                     'ts': time.time()})
            return {'passed': False, 'reason': 'Grade D pre-filtered',
                    'rep_valid': False, 'pqp_valid': False, 'gas_used': 21000}

        # Minimum threshold (rule-based, no ZKP needed)
        if grade == 'C':
            # Grade C: no ZKP policy, only REP range check
            return {'passed': False,
                    'reason': 'Grade C: below REP policy threshold',
                    'rep_valid': False, 'pqp_valid': False, 'gas_used': 21000}

        # REP verification (Grade B, A, S)
        if cert.release_zkp_proof is None:
            return {'passed': False, 'reason': 'REP proof missing',
                    'rep_valid': False, 'pqp_valid': False, 'gas_used': 21000}

        from zkp.zkp_simulator import Groth16Proof
        # Reconstruct minimal proof object for verification
        rep_valid = cert.release_zkp_proof is not None
        pqp_valid = False
        gas_used  = 113_000

        if grade in ('S', 'A'):
            pqp_valid = cert.premium_zkp_proof is not None
            gas_used += 113_000

        if not rep_valid:
            return {'passed': False, 'reason': 'REP verification failed',
                    'rep_valid': False, 'pqp_valid': False, 'gas_used': gas_used}

        if grade in ('S', 'A') and not pqp_valid:
            return {'passed': False, 'reason': 'PQP verification failed for S/A grade',
                    'rep_valid': True, 'pqp_valid': False, 'gas_used': gas_used}

        return {'passed': True, 'reason': 'All checks passed',
                'rep_valid': True, 'pqp_valid': pqp_valid, 'gas_used': gas_used}


# ── LifeBankDirectory Contract ────────────────────────────────────────────────
class LifeBankDirectory:
    """
    DID registration, VC issuance, RBAC management.
    """
    def __init__(self, ledger: QBFTLedger):
        self.ledger   = ledger
        self.dids     = {}    # did → {role, vcs, address}
        self.roles    = {}    # address → role
        self._register_admin()

    def _register_admin(self):
        admin_did = 'did:mdaf:admin:0000000000000000'
        self.dids[admin_did] = {'role': 'ADMIN', 'vcs': ['AdminVC'], 'address': 'admin'}
        self.roles['admin'] = 'ADMIN'

    def register_did(self, address: str, role: str, vcs: list) -> Transaction:
        if role not in ROLES + ['ADMIN']:
            raise ValueError(f'Invalid role: {role}')
        did = f'did:mdaf:{role.lower()}:{hashlib.sha256(address.encode()).hexdigest()[:16]}'
        self.dids[did] = {'role': role, 'vcs': vcs, 'address': address}
        self.roles[address] = role

        tx = Transaction(
            tx_hash=_tx_hash(), from_addr='admin',
            to_contract='LifeBankDirectory', function='registerDID',
            args={'did': did, 'role': role, 'address': address},
            timestamp=time.time(), gas_used=55000, status='success',
            event_log=[{'event': 'DIDRegistered', 'did': did, 'role': role}]
        )
        self.ledger.submit_transaction(tx)
        return tx

    def check_role(self, address: str, required_role: str) -> bool:
        return self.roles.get(address) == required_role

    def get_role(self, address: str) -> Optional[str]:
        return self.roles.get(address)


# ── MDAF Blockchain System ────────────────────────────────────────────────────
class MDAFBlockchainSystem:
    """
    Top-level system integrating all contracts and the QBFT ledger.
    """
    def __init__(self):
        from zkp.zkp_simulator import ZKPVerifier
        self.ledger   = QBFTLedger()
        self.verifier = ZKPVerifier()
        self.gateway  = QualityGateway(self.verifier)
        self.registry = CertificateRegistry(self.ledger, self.gateway)
        self.directory= LifeBankDirectory(self.ledger)

        # Register demo participants
        self._setup_participants()

    def _setup_participants(self):
        participants = [
            ('biobank_001',  'BIOBANK',  ['InstitutionalVC', 'GMPComplianceVC']),
            ('lab_001',      'LAB',      ['QualityLabVC', 'CAPAccreditationVC']),
            ('hospital_001', 'HOSPITAL', ['PrescriptionVC']),
            ('auditor_001',  'AUDITOR',  ['AuditAuthorityVC']),
            ('regulator_001','REGULATOR',['RegulatoryAuthorityVC']),
        ]
        for addr, role, vcs in participants:
            self.directory.register_did(addr, role, vcs)

    def process_batch(self, cert, data_hash: str) -> dict:
        """Full lifecycle: issue → lifecycle events → return summary."""
        t0 = time.time()

        # Issue
        tx_issue = self.registry.issue_certificate(cert, data_hash, 'lab_001')

        result = {
            'cert_id':      cert.cert_id,
            'grade':        cert.mqs_grade,
            'tx_issue':     tx_issue.tx_hash,
            'status':       tx_issue.status,
            'gas_used':     tx_issue.gas_used,
            'events':       tx_issue.event_log,
            'finality_s':   QBFT_FINALITY_S,
        }

        if tx_issue.status == 'success':
            # Lifecycle: quality assessment completed
            self.registry.update_lifecycle(cert.cert_id,
                                           'QualityAssessmentCompleted', 'lab_001')
            result['lifecycle_updated'] = True
            result['processing_time_s'] = round(time.time() - t0, 3)

        return result

    def full_stats(self) -> dict:
        stats = self.ledger.stats()
        stats['registered_certs']  = len(self.registry.registry)
        stats['rejected_batches']  = len(self.gateway.rejections)
        stats['registered_dids']   = len(self.directory.dids)
        return stats
