# -*- coding: utf-8 -*-
"""
test_pipeline.py — End-to-end integration test for the full MDAF pipeline.
Tests: data generation → model training → evaluation →
       ZKP proof → blockchain → digital twin matching
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np, time

from ai_engine.dataset   import generate_synthetic_data, build_dataloaders
from ai_engine.model     import MDAFModel
from ai_engine.train     import train
from ai_engine.evaluate  import evaluate_test, run_baselines, run_ablation
from ai_engine.digital_twin import generate_demo_patients_and_batches, rank_batches_for_patient
from zkp.zkp_simulator   import ZKPProver, run_zkp_benchmark, sha256_hash
from zkp.certificate     import build_certificate
from blockchain.besu_4node.besu_adapter import MDAFBlockchainAdapter as MDAFBlockchainSystem


def test_full_pipeline():
    print('='*70)
    print('  MDAF End-to-End Pipeline Test')
    print('='*70)

    t_start = time.time()
    device = torch.device('cpu')

    # ──────────────────────────────────────────────────────────
    # Step 1: Data generation
    # ──────────────────────────────────────────────────────────
    print('\n[Step 1] Generating synthetic data...')
    train_loader, val_loader, test_loader, scalers, df, cnn, splits = \
        build_dataloaders(data_dir='data', batch_size=64)

    # Grade distribution check
    grade_dist = df['grade_str'].value_counts().to_dict()
    assert len(df) == 2000, f'Expected 2000 samples, got {len(df)}'
    assert set(grade_dist.keys()).issubset({'S','A','B','C','D'})
    print(f'  ✓ Dataset: {len(df)} samples | Grades: {grade_dist}')

    # ──────────────────────────────────────────────────────────
    # Step 2: Model training (short run for test)
    # ──────────────────────────────────────────────────────────
    print('\n[Step 2] Training MDAFModel...')
    model = MDAFModel(dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model parameters: {n_params:,}')

    model, history = train(
        model, train_loader, val_loader, device,
        epochs=15, lr=1e-3, patience=5,
        checkpoint_dir='checkpoints',
    )
    last_val_r2 = history[-1]['val']['r2']
    print(f'  ✓ Training complete | Final val R²={last_val_r2:.3f}')

    # ──────────────────────────────────────────────────────────
    # Step 3: Evaluation
    # ──────────────────────────────────────────────────────────
    print('\n[Step 3] Evaluating on test set...')
    metrics, embeddings, mqs_preds, grade_trues = evaluate_test(model, test_loader, device)

    assert metrics['R2'] > 0.3,        f'R² too low: {metrics["R2"]}'
    assert metrics['AUROC'] > 0.6,     f'AUROC too low: {metrics["AUROC"]}'
    assert metrics['Macro_F1_5class'] > 0.3, f'F1 too low'
    print(f'  ✓ R²={metrics["R2"]}  F1={metrics["Macro_F1_5class"]}  AUROC={metrics["AUROC"]}')

    # ──────────────────────────────────────────────────────────
    # Step 4: Baselines
    # ──────────────────────────────────────────────────────────
    print('\n[Step 4] Running baseline comparison...')
    baseline_results = run_baselines(df, splits)
    assert 'Linear/Logistic' in baseline_results
    assert 'GradientBoosting (XGBoost-style)' in baseline_results
    print('  ✓ Baselines completed')

    # ──────────────────────────────────────────────────────────
    # Step 5: Ablation (quick, 10 epochs)
    # ──────────────────────────────────────────────────────────
    print('\n[Step 5] Running ablation study (quick mode)...')
    abl_results = run_ablation(train_loader, val_loader, test_loader,
                               device, epochs=10)
    assert len(abl_results) == 6
    print('  ✓ Ablation study completed')

    # ──────────────────────────────────────────────────────────
    # Step 6: ZKP proof generation
    # ──────────────────────────────────────────────────────────
    print('\n[Step 6] ZKP proof generation and benchmark...')
    prover = ZKPProver()
    zkp_results = run_zkp_benchmark(df, n=30)

    assert float(zkp_results['REP_success_rate'].rstrip('%')) > 0
    print(f'  ✓ REP success: {zkp_results["REP_success_rate"]}')
    print(f'  ✓ REP prove: {zkp_results["REP_mean_prove_s"]}s | verify: {zkp_results["REP_verify_ms"]}ms')

    # ──────────────────────────────────────────────────────────
    # Step 7: Certificate issuance
    # ──────────────────────────────────────────────────────────
    print('\n[Step 7] Building digital certificates...')
    certificates_built = 0
    sample_certs = []

    # Take top 10 high-MQS batches
    top_batches = df.nlargest(10, 'MQS')
    for _, row in top_batches.iterrows():
        grade = row['grade_str']
        mqs   = float(row['MQS'])
        raw_data = {'batch_id': str(row.name), 'mqs': mqs, 'viability': float(row['viability'])}
        data_hash = sha256_hash(raw_data)
        cert_id_tmp = f'CERT-TEST-{row.name:04d}'

        from zkp.zkp_simulator import ZKPWitness
        witness = ZKPWitness(
            mqs=mqs, viability=float(row['viability']),
            passage=int(row['passage']), sterility_pass=True,
            raw_data_hash=data_hash, markers_valid=float(row['cd90_fold'])>=100,
        )
        proof_rep, rep_ok, _ = prover.generate_rep_proof(witness, cert_id_tmp, data_hash)
        proof_pqp = None
        if grade in ('S', 'A') and rep_ok:
            proof_pqp, _, _ = prover.generate_pqp_proof(witness, cert_id_tmp, data_hash)

        on_chain, off_chain, cert_id, dh = build_certificate(
            row, mqs, grade, proof_rep, proof_pqp
        )
        sample_certs.append((on_chain, off_chain, cert_id, dh, row))
        certificates_built += 1

    assert certificates_built == 10
    print(f'  ✓ {certificates_built} certificates built')

    # ──────────────────────────────────────────────────────────
    # Step 8: Blockchain issuance
    # ──────────────────────────────────────────────────────────
    print('\n[Step 8] Issuing certificates to QBFT blockchain...')
    system = MDAFBlockchainSystem(prefer_real=True)
    mode   = 'real_besu_4node' if system._real else 'simulation'
    print(f'  Blockchain mode: {mode}')
    issued = 0; rejected = 0

    for on_chain, off_chain, cert_id, data_hash, row in sample_certs:
        result = system.process_batch(on_chain, data_hash)
        if result['status'] == 'success':
            issued += 1
            # Record lifecycle
            system.registry.update_lifecycle(cert_id, 'ClinicalReleaseRequested', 'lab_001')
        else:
            rejected += 1

    chain_stats = system.full_stats()
    print(f'  ✓ Issued: {issued} | Rejected: {rejected}')
    print(f'  ✓ Blockchain: {chain_stats["blocks"]} blocks | '
          f'{chain_stats["transactions"]} txs | '
          f'finality={chain_stats["finality_s"]}s')
    assert issued > 0, 'No certificates were issued successfully'

    # ──────────────────────────────────────────────────────────
    # Step 9: Digital twin matching
    # ──────────────────────────────────────────────────────────
    print('\n[Step 9] Digital twin patient-batch matching...')
    patients, batches = generate_demo_patients_and_batches(df)
    all_match_scores = []

    for patient in patients:
        ranked = rank_batches_for_patient(patient, batches, top_k=3)
        assert len(ranked) > 0
        top_score = ranked[0]['match_score']
        all_match_scores.append(top_score)
        # Record match on blockchain for issued certs
        for cert_info in sample_certs[:2]:
            on_chain, _, cert_id, _, _ = cert_info
            if system.registry.query(cert_id):
                system.registry.record_twin_match(cert_id, top_score,
                                                   patient.patient_id, 'lab_001')

    mean_score = np.mean(all_match_scores)
    assert 0 < mean_score <= 100
    print(f'  ✓ Matching complete | Mean top-1 score: {mean_score:.1f}')

    # ──────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print('\n' + '='*70)
    print('  MDAF End-to-End Pipeline: ALL TESTS PASSED ✓')
    print('='*70)
    print(f'\n  Summary:')
    print(f'  ├─ Dataset:        n=2,000 | 5 grades')
    print(f'  ├─ Model:          R²={metrics["R2"]}  F1={metrics["Macro_F1_5class"]}  AUROC={metrics["AUROC"]}')
    print(f'  ├─ ZKP:            REP={zkp_results["REP_mean_prove_s"]}s / {zkp_results["REP_verify_ms"]}ms')
    print(f'  ├─ Blockchain:     {chain_stats["blocks"]} blocks | {issued} certs issued')
    print(f'  ├─ Digital Twin:   Mean score={mean_score:.1f}')
    print(f'  └─ Total time:     {elapsed:.1f}s')
    print()

    return {
        'metrics':     metrics,
        'zkp':         zkp_results,
        'blockchain':  chain_stats,
        'twin_score':  mean_score,
    }


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_full_pipeline()
