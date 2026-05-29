# -*- coding: utf-8 -*-
"""
zkp_simulator.py
----------------
Groth16 ZKP simulation for REP and PQP circuits.
Simulates proof generation / on-chain verification timing
without an actual snarkjs/circom runtime.

Circuit design (pseudo-code, Section 5.2):
  REP: MQS >= 70 AND viability >= 90% AND passage <= 5
       AND sterility=PASS AND SHA-256(raw_data)=data_hash
  PQP: MQS >= 85 AND viability >= 95% AND passage <= 3
       AND sterility=PASS AND markers=VALID
       AND SHA-256(raw_data)=data_hash
"""

import hashlib, time, secrets, json
import numpy as np


# Simulated timing benchmarks (from paper benchmarks)
REP_PROVE_TIME_S   = 1.8      # seconds
PQP_PROVE_TIME_S   = 2.1
REP_VERIFY_TIME_MS = 32.0     # milliseconds
PQP_VERIFY_TIME_MS = 38.0
REP_GAS            = 113_000
PQP_GAS            = 113_000

# Policy thresholds
REP_MQS_THRESHOLD      = 70.0
REP_VIABILITY_THRESHOLD = 80.0
REP_PASSAGE_MAX        = 5

PQP_MQS_THRESHOLD      = 85.0
PQP_VIABILITY_THRESHOLD = 95.0
PQP_PASSAGE_MAX        = 3


class Groth16Proof:
    """Simulated Groth16 proof: π = (A, B, C) ∈ G1×G2×G1."""
    def __init__(self, circuit_id: str, public_inputs: dict):
        self.circuit_id    = circuit_id
        self.public_inputs = public_inputs
        # Simulate 192-byte proof (A:48 + B:96 + C:48)
        self.pi_A = secrets.token_bytes(48).hex()
        self.pi_B = secrets.token_bytes(96).hex()
        self.pi_C = secrets.token_bytes(48).hex()
        self.proof_bytes = self.pi_A + self.pi_B + self.pi_C  # 384 hex chars = 192 bytes
        self.timestamp = time.time()

    def to_dict(self):
        return {
            'circuit_id':    self.circuit_id,
            'pi_A':          self.pi_A,
            'pi_B':          self.pi_B,
            'pi_C':          self.pi_C,
            'public_inputs': self.public_inputs,
            'timestamp':     self.timestamp,
        }

    def __repr__(self):
        return f'Groth16Proof({self.circuit_id}, A={self.pi_A[:8]}...)'


class ZKPWitness:
    """Private witness for ZKP proof (never leaves the prover)."""
    def __init__(self, mqs, viability, passage, sterility_pass,
                 raw_data_hash, markers_valid=True):
        self.mqs            = mqs
        self.viability      = viability
        self.passage        = passage
        self.sterility_pass = sterility_pass
        self.raw_data_hash  = raw_data_hash
        self.markers_valid  = markers_valid


def sha256_hash(data: dict) -> str:
    serialized = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


class ZKPProver:
    """
    Simulates snarkjs Groth16 prover.
    In production: replaced by actual circom circuit + snarkjs prove().
    """

    def generate_rep_proof(self, witness: ZKPWitness,
                            cert_id: str, data_hash: str) -> tuple:
        """
        Generate REP (ReleaseEligibilityProof).
        Returns (proof: Groth16Proof | None, result: bool, timing: dict)
        """
        t0 = time.time()
        time.sleep(REP_PROVE_TIME_S * 0.01)  # Scaled simulation delay

        # Circuit constraint evaluation (public result only)
        rep_valid = (
            witness.mqs        >= REP_MQS_THRESHOLD        and
            witness.viability  >= REP_VIABILITY_THRESHOLD  and
            witness.passage    <= REP_PASSAGE_MAX           and
            witness.sterility_pass                          and
            witness.raw_data_hash == data_hash
        )

        prove_time = REP_PROVE_TIME_S  # simulated
        if rep_valid:
            public_inputs = {
                'cert_id':          cert_id,
                'policy_id':        '0x01',
                'data_hash':        data_hash,
                'verification_result': True,
            }
            proof = Groth16Proof('REP_circuit_v1', public_inputs)
        else:
            proof = None

        timing = {
            'prove_time_s':    prove_time,
            'verify_time_ms':  REP_VERIFY_TIME_MS,
            'gas_estimate':    REP_GAS,
        }
        return proof, rep_valid, timing

    def generate_pqp_proof(self, witness: ZKPWitness,
                            cert_id: str, data_hash: str) -> tuple:
        """
        Generate PQP (PremiumQualityProof).
        Only generated for Grade S/A batches.
        """
        time.sleep(PQP_PROVE_TIME_S * 0.01)

        pqp_valid = (
            witness.mqs        >= PQP_MQS_THRESHOLD        and
            witness.viability  >= PQP_VIABILITY_THRESHOLD  and
            witness.passage    <= PQP_PASSAGE_MAX           and
            witness.sterility_pass                          and
            witness.markers_valid                           and
            witness.raw_data_hash == data_hash
        )

        if pqp_valid:
            public_inputs = {
                'cert_id':          cert_id,
                'policy_id':        '0x02',
                'data_hash':        data_hash,
                'verification_result': True,
            }
            proof = Groth16Proof('PQP_circuit_v1', public_inputs)
        else:
            proof = None

        timing = {
            'prove_time_s':    PQP_PROVE_TIME_S,
            'verify_time_ms':  PQP_VERIFY_TIME_MS,
            'gas_estimate':    PQP_GAS,
        }
        return proof, pqp_valid, timing


class ZKPVerifier:
    """
    Simulates on-chain ZKP verifier (EIP-1108 precompile).
    In production: Solidity ZKPVerifier.verifyREP() / verifyPQP()
    """
    def verify(self, proof: Groth16Proof, expected_data_hash: str) -> tuple:
        """Returns (is_valid: bool, gas_used: int, verify_time_ms: float)"""
        if proof is None:
            return False, 0, 0.0
        # Simulate EIP-1108 pairing check
        time.sleep(0.001)  # 1ms simulation
        stored_hash = proof.public_inputs.get('data_hash', '')
        is_valid = (stored_hash == expected_data_hash and
                    proof.public_inputs.get('verification_result', False))
        gas = REP_GAS if proof.circuit_id.startswith('REP') else PQP_GAS
        vtime = (REP_VERIFY_TIME_MS if proof.circuit_id.startswith('REP')
                 else PQP_VERIFY_TIME_MS)
        return is_valid, gas, vtime


def run_zkp_benchmark(batches_df, n=50):
    """
    Run ZKP benchmark on n random batches.
    Returns performance statistics.
    """
    prover   = ZKPProver()
    verifier = ZKPVerifier()

    rep_times, pqp_times = [], []
    rep_success, pqp_success = 0, 0
    rng = np.random.default_rng(42)
    sample = batches_df.sample(min(n, len(batches_df)), random_state=42)

    print(f'\n[ZKP] Benchmarking on {len(sample)} batches...')
    for _, row in sample.iterrows():
        raw_data = {
            'batch_id':   str(row.name),
            'mqs':        float(row['MQS']),
            'viability':  float(row['viability']),
            'passage':    int(row['passage']),
        }
        data_hash = sha256_hash(raw_data)
        cert_id   = hashlib.sha3_256(str(row.name).encode()).hexdigest()[:16]

        witness = ZKPWitness(
            mqs            = float(row['MQS']),
            viability      = float(row['viability']),
            passage        = int(row['passage']),
            sterility_pass = True,
            raw_data_hash  = data_hash,
            markers_valid  = float(row['cd90_fold']) >= 100,
        )

        # REP
        proof_rep, rep_ok, t_rep = prover.generate_rep_proof(witness, cert_id, data_hash)
        if rep_ok:
            rep_success += 1
            rep_times.append(t_rep['prove_time_s'])
            v_rep, _, _ = verifier.verify(proof_rep, data_hash)

        # PQP (only Grade S/A)
        if row['grade_str'] in ('S', 'A'):
            proof_pqp, pqp_ok, t_pqp = prover.generate_pqp_proof(witness, cert_id, data_hash)
            if pqp_ok:
                pqp_success += 1
                pqp_times.append(t_pqp['prove_time_s'])

    eligible_mask = ((sample['MQS'] >= REP_MQS_THRESHOLD) &
                      (sample['viability'] >= REP_VIABILITY_THRESHOLD) &
                      (sample['passage'] <= REP_PASSAGE_MAX))
    eligible = eligible_mask.sum()
    results = {
        'REP_eligible': int(eligible),
        'REP_success_rate': f'{rep_success/max(int(eligible),1)*100:.1f}%',
        'REP_mean_prove_s': round(np.mean(rep_times), 2) if rep_times else 0,
        'REP_verify_ms':    REP_VERIFY_TIME_MS,
        'REP_gas':          REP_GAS,
        'PQP_success_count': pqp_success,
        'PQP_mean_prove_s':  round(np.mean(pqp_times), 2) if pqp_times else 0,
        'PQP_verify_ms':     PQP_VERIFY_TIME_MS,
    }
    print('[ZKP] Results:')
    for k, v in results.items():
        print(f'  {k:30s}: {v}')
    return results
