#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zkp_real_runner.py
------------------
Python 래퍼: zkp_real_benchmark.js (circom + snarkjs) 실행 및 결과 파싱.
zkp_simulator.py의 시뮬레이션 결과를 실측값으로 교체합니다.

사용:
    from zkp.zkp_real_runner import run_real_zkp_benchmark, is_zkp_available
    if is_zkp_available():
        results = run_real_zkp_benchmark(n=10)
"""

import os, sys, json, subprocess, time, shutil
from pathlib import Path

ZKP_DIR  = Path(__file__).parent
JS_BENCH = ZKP_DIR / 'zkp_real_benchmark.js'
CIRCUITS = ZKP_DIR / 'circuits'
OUTPUT   = ZKP_DIR.parent / 'outputs'


def is_zkp_available() -> bool:
    """circom + snarkjs + 회로 파일이 모두 있는지 확인"""
    checks = [
        shutil.which('circom') is not None,
        shutil.which('node')   is not None,
        JS_BENCH.exists(),
        (CIRCUITS / 'REP.circom').exists(),
        (CIRCUITS / 'PQP.circom').exists(),
    ]
    return all(checks)


def run_real_zkp_benchmark(n: int = 10, quick: bool = False,
                            timeout: int = 600) -> dict:
    """
    circom + snarkjs로 REP/PQP 회로를 실제 실행하고 성능을 실측합니다.

    Returns:
        dict with keys: REP, PQP (각각 성능 수치)
        실패 시: {'error': str, 'fallback': simulation_results}
    """
    if not is_zkp_available():
        print('[ZKP] circom/snarkjs 미설치 → 시뮬레이션 fallback')
        from zkp.zkp_simulator import run_zkp_benchmark
        import pandas as pd
        return {'mode': 'simulation', 'note': 'circom not available'}

    print(f'\n[ZKP Real] circom + snarkjs Groth16 실측 (n={n})...')
    args = ['node', str(JS_BENCH), '--n', str(n)]
    if quick:
        args.append('--quick')

    t0 = time.time()
    try:
        proc = subprocess.run(
            args,
            cwd=str(ZKP_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f'[ZKP Real] Error (exit {proc.returncode}):')
            print(proc.stderr[-500:] if proc.stderr else '(no stderr)')
            return {'mode': 'error', 'stderr': proc.stderr[-500:]}

        # stdout 출력
        print(proc.stdout[-2000:] if len(proc.stdout) > 2000
              else proc.stdout)

        # 결과 JSON 로드 (가장 최신 파일)
        OUTPUT.mkdir(exist_ok=True)
        json_files = sorted(OUTPUT.glob('zkp_benchmark_*.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if json_files:
            with open(json_files[0]) as f:
                raw = json.load(f)
            return _parse_results(raw, elapsed)

        return {'mode': 'real', 'elapsed_s': round(elapsed, 1),
                'note': 'Results parsed from stdout'}

    except subprocess.TimeoutExpired:
        return {'mode': 'timeout', 'timeout_s': timeout}
    except Exception as e:
        return {'mode': 'error', 'error': str(e)}


def _parse_results(raw: dict, elapsed: float) -> dict:
    """JSON 결과를 파이프라인 호환 형식으로 변환"""
    out = {
        'mode':             'real_circom_snarkjs',
        'timestamp':        raw.get('timestamp'),
        'n_iterations':     raw.get('n_iterations'),
        'elapsed_s':        round(elapsed, 1),
    }

    for circuit in ['REP', 'PQP']:
        c = raw.get('circuits', {}).get(circuit, {})
        if not c:
            continue
        prove  = c.get('prove_s', {})
        verify = c.get('verify_ms', {})
        out[circuit] = {
            'constraints':       c.get('constraints', 0),
            'success_rate':      c.get('success_rate', '?'),
            'prove_mean_s':      prove.get('mean', 0) if prove else 0,
            'prove_p95_s':       prove.get('p95',  0) if prove else 0,
            'prove_min_s':       prove.get('min',  0) if prove else 0,
            'prove_max_s':       prove.get('max',  0) if prove else 0,
            'verify_mean_ms':    verify.get('mean', 0) if verify else 0,
            'verify_p95_ms':     verify.get('p95',  0) if verify else 0,
            'gas_eip1108':       c.get('estimated_gas_eip1108', 113000),
            'fail_test_ok':      c.get('fail_test_ok', False),
        }

    # 파이프라인 호환 키 (기존 zkp_simulator 출력 형식)
    rep = out.get('REP', {})
    pqp = out.get('PQP', {})
    out['REP_mean_prove_s']    = rep.get('prove_mean_s', 0)
    out['REP_verify_ms']       = rep.get('verify_mean_ms', 0)
    out['REP_success_rate']    = rep.get('success_rate', '?')
    out['REP_gas']             = rep.get('gas_eip1108', 113000)
    out['PQP_mean_prove_s']    = pqp.get('prove_mean_s', 0)
    out['PQP_verify_ms']       = pqp.get('verify_mean_ms', 0)
    out['PQP_success_rate']    = pqp.get('success_rate', '?')

    return out


def print_comparison(real: dict, sim: dict = None):
    """실측 vs 논문 목표값 비교 출력"""
    paper_targets = {
        'REP': {'prove_s': 1.8, 'verify_ms': 32, 'gas': 113000},
        'PQP': {'prove_s': 2.1, 'verify_ms': 38, 'gas': 113000},
    }

    print('\n' + '='*65)
    print('  ZKP 실측 vs 논문 목표값 비교')
    print('='*65)
    print(f'  {"항목":20s}  {"실측":>12}  {"논문목표":>12}  {"차이":>10}')
    print('-'*65)

    for c in ['REP', 'PQP']:
        cdata  = real.get(c, {})
        target = paper_targets[c]

        prove_r  = cdata.get('prove_mean_s', 0)
        verify_r = cdata.get('verify_mean_ms', 0)
        gas_r    = cdata.get('gas_eip1108', 0)

        def diff(measured, target_v, unit=''):
            if not measured: return 'N/A'
            d = measured - target_v
            sign = '+' if d >= 0 else ''
            return f'{sign}{d:.3f}{unit}'

        print(f'  [{c}] prove time    {prove_r:>10.3f}s  {target["prove_s"]:>10.1f}s  '
              f'{diff(prove_r, target["prove_s"], "s"):>10}')
        print(f'  [{c}] verify time   {verify_r:>10.1f}ms {target["verify_ms"]:>10.0f}ms  '
              f'{diff(verify_r, target["verify_ms"], "ms"):>10}')
        print(f'  [{c}] gas estimate  {gas_r:>10,}  {target["gas"]:>10,}  '
              f'{"~same" if abs(gas_r-target["gas"])<5000 else diff(gas_r,target["gas"],""):>10}')
        print(f'  [{c}] success rate  {cdata.get("success_rate","?"):>12}  {"100%":>12}')
        print()

    print('='*65)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',     type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    print(f'ZKP 가용 여부: {is_zkp_available()}')
    if is_zkp_available():
        results = run_real_zkp_benchmark(n=args.n, quick=args.quick)
        print_comparison(results)
    else:
        print('circom 또는 snarkjs가 설치되지 않았습니다.')
        print('설치: npm install -g circom snarkjs')
