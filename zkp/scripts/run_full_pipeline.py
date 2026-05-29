# -*- coding: utf-8 -*-
"""
run_full_pipeline.py
--------------------
Single entry point: runs the complete MDAF pipeline end-to-end.
  data generation → training → evaluation → ZKP → blockchain → digital twin
Usage:
    cd mdaf
    python scripts/run_full_pipeline.py [--epochs 60] [--batch_size 64] [--device cpu]
"""

import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from ai_engine.dataset    import build_dataloaders
from ai_engine.model      import MDAFModel
from ai_engine.train      import train
from ai_engine.evaluate   import evaluate_test, run_baselines, run_ablation
from ai_engine.shap_analysis import run_shap_analysis
from ai_engine.digital_twin  import (generate_demo_patients_and_batches,
                                      rank_batches_for_patient)
from zkp.zkp_simulator    import ZKPProver, ZKPWitness, run_zkp_benchmark, sha256_hash
from zkp.certificate      import build_certificate
from zkp.zkp_real_runner  import (run_real_zkp_benchmark,
                                   is_zkp_available, print_comparison)
# ── Blockchain: real Besu 4-node if Docker running, else simulation ──────────
# To use real Besu:  cd blockchain/besu_4node && python scripts/setup_network.py
#                    docker compose up -d   (wait ~20s)
# Adapter auto-detects localhost:8545 — no code change needed.
from blockchain.besu_4node.besu_adapter import MDAFBlockchainAdapter as MDAFBlockchainSystem


def main(args):
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f'\n{"="*70}')
    print(f'  MDAF Full Pipeline  |  device={device}  epochs={args.epochs}')
    if args.run_qbft_bench:
        print(f'  QBFT benchmark: ENABLED (will run after blockchain step)')
    print(f'{"="*70}')

    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────
    print('\n── 1. Data Generation ─────────────────────────────────')
    train_loader, val_loader, test_loader, scalers, df, cnn, splits = \
        build_dataloaders(data_dir='data', batch_size=args.batch_size)

    # ── 2. Train ───────────────────────────────────────────────
    print('\n── 2. Model Training ──────────────────────────────────')
    model = MDAFModel(dropout=0.2).to(device)
    model, history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=1e-3, patience=12,
        checkpoint_dir='checkpoints',
    )

    # ── 3. Evaluate ────────────────────────────────────────────
    print('\n── 3. Test Set Evaluation ─────────────────────────────')
    metrics, embeddings, mqs_preds, grade_trues = evaluate_test(model, test_loader, device)

    # ── 4. Baselines ───────────────────────────────────────────
    print('\n── 4. Baseline Comparison ─────────────────────────────')
    baselines = run_baselines(df, splits)

    # ── 5. Ablation ────────────────────────────────────────────
    print('\n── 5. Ablation Study ──────────────────────────────────')
    ablation = run_ablation(train_loader, val_loader, test_loader, device, epochs=20)

    # ── 6. SHAP ────────────────────────────────────────────────
    if args.run_shap:
        print('\n── 6. SHAP Analysis ───────────────────────────────────')
        from ai_engine.dataset import MSCDataset
        test_idx = splits['test']
        import pandas as pd
        ds_test_df = df.iloc[test_idx].reset_index(drop=True)
        from ai_engine.dataset import MSCDataset as MDS
        ds_test = MDS(ds_test_df, cnn[test_idx], scalers=scalers, fit_scalers=False)
        shap_vals, modality_pct, feature_contrib = run_shap_analysis(
            model, ds_test, device, n_background=50, n_explain=100)
    else:
        modality_pct = {}
        print('\n── 6. SHAP Analysis (skipped, use --run_shap to enable) ──')

    # ── 7. ZKP Benchmark ─────────────────────────────────────────
    print('\n── 7. ZKP Benchmark ───────────────────────────────────')
    if is_zkp_available():
        print('  circom + snarkjs 실측 모드')
        zkp_results = run_real_zkp_benchmark(n=5, quick=True)
        print_comparison(zkp_results)
    else:
        print('  시뮬레이션 모드 (circom 미설치)')
        zkp_results = run_zkp_benchmark(df, n=50)

    # ── 8. Certificate + Blockchain ────────────────────────────
    print('\n── 8. Certificate Issuance → Blockchain ───────────────')
    prover = ZKPProver()
    system = MDAFBlockchainSystem(prefer_real=True)   # real Besu if available
    issued = 0
    sample_certs = []

    for _, row in df.nlargest(30, 'MQS').iterrows():
        grade = row['grade_str']
        mqs   = float(row['MQS'])
        raw_data  = {'batch_id': str(row.name), 'mqs': mqs, 'viability': float(row['viability'])}
        data_hash = sha256_hash(raw_data)
        cert_id_t = f'CERT-{row.name:06d}'
        witness = ZKPWitness(
            mqs=mqs, viability=float(row['viability']),
            passage=int(row['passage']), sterility_pass=True,
            raw_data_hash=data_hash, markers_valid=float(row['cd90_fold'])>=100,
        )
        proof_rep, rep_ok, _ = prover.generate_rep_proof(witness, cert_id_t, data_hash)
        proof_pqp = None
        if grade in ('S','A') and rep_ok:
            proof_pqp, _, _ = prover.generate_pqp_proof(witness, cert_id_t, data_hash)

        on_chain, off_chain, cert_id, dh = build_certificate(row, mqs, grade,
                                                               proof_rep, proof_pqp)
        result = system.process_batch(on_chain, dh)
        if result['status'] == 'success':
            issued += 1
            sample_certs.append((cert_id, mqs, grade))

    chain_stats = system.full_stats()
    blockchain_mode = chain_stats.get('mode', 'simulation')
    fin_label = (f"{chain_stats.get('mean_finality_s','?')}s (measured)"
                 if blockchain_mode == 'real_besu_4node'
                 else f"{chain_stats.get('finality_s', 1.2)}s (simulated)")
    print(f'\n  Issued: {issued}/30 | Mode: {blockchain_mode} | {chain_stats}')

    # ── 8.5 QBFT 실측 벤치마크 (Docker 기동 중 + --run_qbft_bench 시) ──
    qbft_bench_results = None
    if args.run_qbft_bench:
        print('\n── 8.5. QBFT Empirical Benchmark ─────────────────────')
        try:
            import sys as _sys
            _besu_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'blockchain', 'besu_4node')
            if _besu_dir not in _sys.path:
                _sys.path.insert(0, _besu_dir)
            from qbft_benchmark import run_benchmarks as _run_qbft
            qbft_bench_results = _run_qbft(quick=False)
            print('  ✓ QBFT benchmark complete')
        except SystemExit:
            print('  ⚠ QBFT benchmark skipped (node not running)')
        except Exception as e:
            print(f'  ⚠ QBFT benchmark error: {e}')
    else:
        print('\n  (QBFT empirical benchmark skipped — use --run_qbft_bench to enable)')

    # ── 9. Digital Twin ────────────────────────────────────────
    print('\n── 9. Digital Twin Matching ───────────────────────────')
    patients, batches = generate_demo_patients_and_batches(df)
    twin_results = []
    for patient in patients:
        ranked = rank_batches_for_patient(patient, batches, top_k=5)
        twin_results.append({'patient': patient.patient_id,
                              'top_match': ranked[0]['match_score'],
                              'tier': ranked[0]['tier']})

    # ── Final report ───────────────────────────────────────────
    elapsed = time.time() - t0
    report = {
        'metrics':         metrics,
        'baselines':       baselines,
        'ablation':        ablation,
        'shap':            modality_pct,
        'zkp':             zkp_results,
        'blockchain':      chain_stats,
        'qbft_benchmark':  qbft_bench_results,
        'twin':            twin_results,
        'elapsed_s':       round(elapsed, 1),
    }

    with open('outputs/pipeline_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f'\n{"="*70}')
    print(f'  MDAF Pipeline Complete  |  Total time: {elapsed:.1f}s')
    print(f'  Report saved → outputs/pipeline_report.json')
    print(f'{"="*70}')
    print(f'\n  Key Results:')
    print(f'  ├─ MQS  R²={metrics["R2"]}  Pearson r={metrics["Pearson_r"]}  RMSE={metrics["RMSE"]}')
    print(f'  ├─ Grade F1(5-cls)={metrics["Macro_F1_5class"]}  F1(4-cls)={metrics["Macro_F1_4class"]}  AUROC={metrics["AUROC"]}')
    if zkp_results.get('mode') == 'real_circom_snarkjs':
        print(f'  ├─ ZKP[실측]  REP={zkp_results.get("REP_mean_prove_s","?")}s / '
              f'{zkp_results.get("REP_verify_ms","?")}ms | '
              f'PQP={zkp_results.get("PQP_mean_prove_s","?")}s / '
              f'{zkp_results.get("PQP_verify_ms","?")}ms')
    else:
        print(f'  ├─ ZKP[시뮬] REP={zkp_results.get("REP_mean_prove_s","?")}s prove / {zkp_results.get("REP_verify_ms","?")}ms verify')
    print(f'  ├─ Chain  [{blockchain_mode}] {chain_stats.get("blocks","?")} blocks | {issued} certs | finality={fin_label}')
    if qbft_bench_results and 'sequential_tps' in qbft_bench_results:
        st = qbft_bench_results['sequential_tps']
        bt = qbft_bench_results.get('burst_tps', {})
        ci = qbft_bench_results.get('cert_issuance', {})
        bi = qbft_bench_results.get('block_interval', {})
        sf = qbft_bench_results.get('single_finality', {})
        print(f'  ├─ QBFT[실측] block={bi.get("mean_interval_s","?")}s | '
              f'finality={sf.get("mean_finality_s","?")}s | '
              f'seq_TPS={st.get("TPS_total","?")} | '
              f'burst_TPS={bt.get("TPS_confirmed","?")} | '
              f'cert_gas={ci.get("mean_gas","?"):,}' if isinstance(ci.get("mean_gas"), int)
              else f'  ├─ QBFT[실측] seq_TPS={st.get("TPS_total","?")} | '
                   f'burst_TPS={bt.get("TPS_confirmed","?")}')
    print(f'  └─ Twin   Mean top-1 score={np.mean([r["top_match"] for r in twin_results]):.1f}')

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDAF Full Pipeline')
    parser.add_argument('--epochs',          type=int,  default=40)
    parser.add_argument('--batch_size',      type=int,  default=64)
    parser.add_argument('--device',          type=str,  default='cpu')
    parser.add_argument('--run_shap',        action='store_true')
    parser.add_argument('--run_qbft_bench',  action='store_true',
                        help='Docker Besu 기동 중일 때 QBFT 실측 벤치마크 실행')
    args = parser.parse_args()

    # Run from mdaf directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(args)
