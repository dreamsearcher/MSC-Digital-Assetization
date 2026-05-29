#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qbft_benchmark.py
=================
실제 Docker 기반 Hyperledger Besu QBFT 4노드 네트워크에서
TPS, 트랜잭션 finality, gas 사용량, 블록 생성 시간을 실측합니다.

측정 항목:
  1. Block production interval   — 실제 QBFT 블록 생성 주기
  2. Single-tx finality          — 트랜잭션 제출 → 온체인 확정까지 시간
  3. Sequential TPS              — 순차 제출 처리량
  4. Burst TPS                   — 병렬 burst 제출 처리량
  5. Certificate issuance timing — storeRecord() 실측
  6. Lifecycle append timing     — appendLifecycle() 실측
  7. Gas usage per op            — 각 operation별 실측 gas
  8. Multi-node consistency      — 4개 노드 블록 동기화 확인

실행 방법:
  cd mdaf/blockchain/besu_4node
  python scripts/setup_network.py          # 최초 1회
  docker compose up -d                     # 노드 기동
  python qbft_benchmark.py                 # 벤치마크 실행
  python qbft_benchmark.py --quick         # 빠른 테스트 (소규모)

출력:
  outputs/qbft_benchmark_<timestamp>.json  — 상세 결과
  outputs/qbft_benchmark_summary.txt       — 논문용 요약
"""

import os, sys, time, json, hashlib, statistics, argparse, datetime, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

# ── 설정 ──────────────────────────────────────────────────────
NODE_URLS = {
    'validator1': 'http://localhost:18545',
    'validator2': 'http://localhost:18547',
    'validator3': 'http://localhost:18548',
    'validator4': 'http://localhost:18549',
}
DEPLOYER_KEY = '0x8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63'
CHAIN_ID     = 1337
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs')

# Solidity 소스 경로
SOL_PATH = os.path.join(os.path.dirname(__file__), 'contracts', 'MDAFBenchmark.sol')

# ── Solidity 컴파일 ───────────────────────────────────────────
def compile_contract() -> Tuple[list, str]:
    """MDAFBenchmark.sol 컴파일 → (ABI, bytecode)"""
    try:
        from solcx import compile_source, install_solc, get_installed_solc_versions
    except ImportError:
        raise RuntimeError("pip install py-solc-x")

    versions = [str(v) for v in get_installed_solc_versions()]
    if '0.8.20' in versions:
        ver = '0.8.20'
    elif versions:
        ver = versions[-1]
    else:
        print("  solc 0.8.0 설치 중...")
        install_solc('0.8.0')
        ver = '0.8.0'

    with open(SOL_PATH, encoding='utf-8') as f:
        source = f.read()

    compiled = compile_source(
        source,
        output_values=['abi', 'bin'],
        solc_version=ver,
        # ★ London EVM 지정: PUSH0(0x5f) 오퍼코드 생성 방지
        # genesis.json이 londonBlock:0 기준이므로 shanghai/paris 타겟 불가
        optimize=True,
        optimize_runs=200,
        evm_version='london',
    )
    _, iface = next(iter(compiled.items()))
    print(f"  컴파일 완료 (solc {ver}, evm=london, bytecode {len(iface['bin'])//2} bytes)")
    return iface['abi'], iface['bin']


# ── web3 연결 헬퍼 ───────────────────────────────────────────
def make_w3(url: str) -> Optional[Web3]:
    try:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 10}))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if w3.is_connected():
            return w3
    except Exception:
        pass
    return None


# ── Gas price 헬퍼 ──────────────────────────────────────────
def get_gas_price(w3: Web3) -> int:
    """노드에서 실제 최소 gas price 조회. 실패 시 단계적으로 증가하며 시도."""
    # 1) eth_gasPrice RPC로 노드 권장값 조회
    try:
        gp = int(w3.eth.gas_price)
        if gp > 0:
            return gp
    except Exception:
        pass
    # 2) eth_minGasPrice (Besu 전용 RPC)
    try:
        result = w3.provider.make_request('eth_minGasPrice', [])
        gp = int(result['result'], 16)
        if gp > 0:
            return gp
    except Exception:
        pass
    # 3) txpool_gasPrice
    try:
        result = w3.provider.make_request('txpool_gasPrice', [])
        gp = int(result.get('result', '0x1'), 16)
        if gp > 0:
            return gp
    except Exception:
        pass
    # 4) 폴백: 1 Gwei
    return 1_000_000_000


# ── 컨트랙트 배포 ────────────────────────────────────────────
def deploy_contract(w3: Web3, abi: list, bytecode: str,
                    account) -> Tuple[object, str]:
    """컨트랙트 배포 → (contract, address)"""
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce    = w3.eth.get_transaction_count(account.address)
    try:
        gas_est = contract.constructor().estimate_gas({'from': account.address})
    except Exception:
        gas_est = 1_500_000   # estimate 실패 시 안전한 고정값

    gas_price = get_gas_price(w3)
    print(f"  gas price: {gas_price:,} wei")
    tx = contract.constructor().build_transaction({
        'from':     account.address,
        'nonce':    nonce,
        'gas':      gas_est + 50_000,
        'gasPrice': gas_price,
        'chainId':  CHAIN_ID,
    })
    signed  = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
    addr    = receipt['contractAddress']
    deployed= w3.eth.contract(address=addr, abi=abi)
    print(f"  배포 완료: {addr}  (gas: {receipt['gasUsed']:,})")
    return deployed, addr


# ═══════════════════════════════════════════════════════════════
# Benchmark 1: 블록 생성 주기 측정
# ═══════════════════════════════════════════════════════════════
def measure_block_interval(w3: Web3, n_blocks: int = 10) -> Dict:
    """실제 QBFT 블록 생성 간격 측정"""
    print(f"\n[1] 블록 생성 주기 측정 ({n_blocks} blocks)...")

    # 기다려서 새 블록 n개 생성 타이밍 측정
    start_block = w3.eth.block_number
    intervals   = []
    timestamps  = []

    prev_time = time.time()
    target    = start_block + 1

    for i in range(n_blocks):
        # 다음 블록 생성 대기
        while w3.eth.block_number < target:
            time.sleep(0.05)
        now = time.time()
        intervals.append(now - prev_time)
        blk = w3.eth.get_block(target)
        timestamps.append(blk['timestamp'])
        prev_time = now
        target   += 1

    # 실제 블록 타임스탬프 기반 간격
    ts_intervals = [timestamps[i+1] - timestamps[i]
                    for i in range(len(timestamps)-1)]

    result = {
        'n_blocks':              n_blocks,
        'wall_clock_intervals':  intervals,
        'block_timestamp_diffs': ts_intervals,
        'mean_interval_s':       round(statistics.mean(intervals), 3),
        'min_interval_s':        round(min(intervals), 3),
        'max_interval_s':        round(max(intervals), 3),
        'stdev_interval_s':      round(statistics.stdev(intervals) if len(intervals)>1 else 0, 3),
        'mean_ts_diff_s':        round(statistics.mean(ts_intervals), 3) if ts_intervals else 0,
    }
    print(f"  mean={result['mean_interval_s']}s  "
          f"min={result['min_interval_s']}s  "
          f"max={result['max_interval_s']}s  "
          f"stdev={result['stdev_interval_s']}s")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 2: 단일 트랜잭션 Finality 측정
# ═══════════════════════════════════════════════════════════════
def measure_single_tx_finality(w3: Web3, contract, account,
                                n: int = 30) -> Dict:
    """트랜잭션 제출 → receipt 확정까지 시간 실측"""
    print(f"\n[2] 단일 트랜잭션 Finality 측정 (n={n})...")

    finalities = []
    gas_useds  = []
    nonce = w3.eth.get_transaction_count(account.address)

    for i in range(n):
        tx = contract.functions.ping().build_transaction({
            'from':     account.address,
            'nonce':    nonce + i,
            'gas':      60_000,
            'gasPrice': get_gas_price(w3),
            'chainId':  CHAIN_ID,
        })
        signed = account.sign_transaction(tx)

        t0      = time.time()
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        elapsed = time.time() - t0

        finalities.append(elapsed)
        gas_useds.append(receipt['gasUsed'])

    result = {
        'n':                   n,
        'operation':           'ping()',
        'finality_list_s':     [round(f, 4) for f in finalities],
        'mean_finality_s':     round(statistics.mean(finalities), 4),
        'median_finality_s':   round(statistics.median(finalities), 4),
        'min_finality_s':      round(min(finalities), 4),
        'max_finality_s':      round(max(finalities), 4),
        'p95_finality_s':      round(sorted(finalities)[int(n*0.95)], 4),
        'stdev_finality_s':    round(statistics.stdev(finalities) if n>1 else 0, 4),
        'mean_gas':            round(statistics.mean(gas_useds)),
    }
    print(f"  mean={result['mean_finality_s']}s  "
          f"median={result['median_finality_s']}s  "
          f"p95={result['p95_finality_s']}s  "
          f"gas={result['mean_gas']:,}")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 3: Sequential TPS
# ═══════════════════════════════════════════════════════════════
def measure_sequential_tps(w3: Web3, contract, account,
                            n: int = 50) -> Dict:
    """순차 제출 → 모두 확정까지 실측 TPS"""
    print(f"\n[3] Sequential TPS 측정 (n={n} txs)...")

    nonce_base = w3.eth.get_transaction_count(account.address)
    tx_hashes  = []

    # 트랜잭션 미리 서명
    signed_txs = []
    for i in range(n):
        tx = contract.functions.ping().build_transaction({
            'from':     account.address,
            'nonce':    nonce_base + i,
            'gas':      60_000,
            'gasPrice': get_gas_price(w3),
            'chainId':  CHAIN_ID,
        })
        signed_txs.append(account.sign_transaction(tx))

    # 순차 제출
    t_start = time.time()
    for stx in signed_txs:
        h = w3.eth.send_raw_transaction(stx.raw_transaction)
        tx_hashes.append(h)
    t_send_done = time.time()

    # 모든 receipt 대기
    confirmed = 0
    block_nums = []
    for h in tx_hashes:
        try:
            r = w3.eth.wait_for_transaction_receipt(h, timeout=60)
            if r['status'] == 1:
                confirmed += 1
                block_nums.append(r['blockNumber'])
        except Exception:
            pass
    t_all_done = time.time()

    total_time  = t_all_done - t_start
    send_time   = t_send_done - t_start
    tps_total   = confirmed / total_time if total_time > 0 else 0
    tps_send    = n / send_time if send_time > 0 else 0
    n_blocks    = len(set(block_nums))

    result = {
        'n_submitted':    n,
        'n_confirmed':    confirmed,
        'total_time_s':   round(total_time, 3),
        'send_time_s':    round(send_time, 3),
        'TPS_total':      round(tps_total, 2),
        'TPS_send':       round(tps_send, 2),
        'blocks_used':    n_blocks,
        'tx_per_block':   round(confirmed / n_blocks, 1) if n_blocks else 0,
    }
    print(f"  confirmed={confirmed}/{n}  time={total_time:.2f}s  "
          f"TPS={tps_total:.2f}  blocks={n_blocks}")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 4: Burst TPS (병렬 제출)
# ═══════════════════════════════════════════════════════════════
def measure_burst_tps(w3: Web3, contract, account,
                      n: int = 100, workers: int = 4) -> Dict:
    """멀티스레드 burst 제출 → TPS 실측"""
    print(f"\n[4] Burst TPS 측정 (n={n}, workers={workers})...")

    nonce_base = w3.eth.get_transaction_count(account.address)

    # 전체 서명 미리 완료
    signed_txs = []
    for i in range(n):
        tx = contract.functions.ping().build_transaction({
            'from':     account.address,
            'nonce':    nonce_base + i,
            'gas':      60_000,
            'gasPrice': get_gas_price(w3),
            'chainId':  CHAIN_ID,
        })
        signed_txs.append(account.sign_transaction(tx))

    # Burst 제출 (ThreadPoolExecutor)
    send_results = []
    lock = threading.Lock()

    def send_one(stx):
        try:
            h = w3.eth.send_raw_transaction(stx.raw_transaction)
            with lock:
                send_results.append(h)
            return True
        except Exception as e:
            return False

    t_burst_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(send_one, stx) for stx in signed_txs]
        for f in as_completed(futures):
            pass
    t_burst_end = time.time()
    burst_time = t_burst_end - t_burst_start

    # Receipt 수집
    confirmed   = 0
    block_nums  = []
    t_wait_start = time.time()
    for h in send_results:
        try:
            r = w3.eth.wait_for_transaction_receipt(h, timeout=90)
            if r['status'] == 1:
                confirmed += 1
                block_nums.append(r['blockNumber'])
        except Exception:
            pass
    t_wait_end  = time.time()
    total_time  = t_wait_end - t_burst_start
    tps_burst   = len(send_results) / burst_time if burst_time > 0 else 0
    tps_total   = confirmed / total_time if total_time > 0 else 0
    n_blocks    = len(set(block_nums))

    result = {
        'n_submitted':       n,
        'n_sent':            len(send_results),
        'n_confirmed':       confirmed,
        'burst_time_s':      round(burst_time, 3),
        'total_time_s':      round(total_time, 3),
        'TPS_burst_send':    round(tps_burst, 2),
        'TPS_confirmed':     round(tps_total, 2),
        'blocks_used':       n_blocks,
        'workers':           workers,
    }
    print(f"  sent={len(send_results)}/{n}  confirmed={confirmed}  "
          f"burst_TPS={tps_burst:.2f}  confirmed_TPS={tps_total:.2f}  "
          f"blocks={n_blocks}")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 5: Certificate Issuance (storeRecord) 실측
# ═══════════════════════════════════════════════════════════════
def measure_cert_issuance(w3: Web3, contract, account,
                          n: int = 20) -> Dict:
    """인증서 발급 (storeRecord) 실측 — finality + gas"""
    print(f"\n[5] Certificate Issuance 실측 (n={n})...")

    finalities = []
    gas_useds  = []
    cert_ids   = []
    nonce_base = w3.eth.get_transaction_count(account.address)

    for i in range(n):
        batch_hash = Web3.keccak(text=f'bench_batch_{i:04d}')
        data_hash  = Web3.keccak(text=f'bench_data_{i:04d}')
        grade      = i % 3      # 0=S, 1=A, 2=B
        passage    = (i % 5) + 1

        tx = contract.functions.storeRecord(
            batch_hash, data_hash, grade, passage
        ).build_transaction({
            'from':     account.address,
            'nonce':    nonce_base + i,
            'gas':      200_000,
            'gasPrice': get_gas_price(w3),
            'chainId':  CHAIN_ID,
        })
        signed = account.sign_transaction(tx)

        t0      = time.time()
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        elapsed = time.time() - t0

        finalities.append(elapsed)
        gas_useds.append(receipt['gasUsed'])

        # certId 추출 (event log)
        try:
            logs = contract.events.RecordStored().process_receipt(receipt)
            if logs:
                cert_ids.append(logs[0]['args']['certId'].hex())
        except Exception:
            cert_ids.append(None)

    result = {
        'n':                   n,
        'operation':           'storeRecord()',
        'mean_finality_s':     round(statistics.mean(finalities), 4),
        'min_finality_s':      round(min(finalities), 4),
        'max_finality_s':      round(max(finalities), 4),
        'p95_finality_s':      round(sorted(finalities)[int(n*0.95)], 4),
        'stdev_finality_s':    round(statistics.stdev(finalities) if n>1 else 0, 4),
        'mean_gas':            round(statistics.mean(gas_useds)),
        'min_gas':             min(gas_useds),
        'max_gas':             max(gas_useds),
        'finality_list_s':     [round(f, 4) for f in finalities],
        'gas_list':            gas_useds,
        'cert_ids':            cert_ids,
    }
    print(f"  mean finality={result['mean_finality_s']}s  "
          f"p95={result['p95_finality_s']}s  "
          f"mean gas={result['mean_gas']:,}")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 6: Lifecycle Event Append 실측
# ═══════════════════════════════════════════════════════════════
def measure_lifecycle_append(w3: Web3, contract, account,
                              cert_ids: List[str], n: int = 20) -> Dict:
    """라이프사이클 이벤트 append 실측"""
    valid_ids = [c for c in cert_ids if c is not None][:n]
    if not valid_ids:
        return {'error': 'No valid cert IDs'}
    print(f"\n[6] Lifecycle Append 실측 (n={len(valid_ids)})...")

    events = [
        'QualityAssessmentCompleted',
        'CertificateIssued',
        'ZKPVerificationCompleted',
        'StorageEventRecorded',
        'TransportEventRecorded',
    ]

    finalities = []
    gas_useds  = []
    nonce_base = w3.eth.get_transaction_count(account.address)

    for i, cid_hex in enumerate(valid_ids):
        cid_bytes  = bytes.fromhex(cid_hex.replace('0x', '').zfill(64))
        event_type = Web3.keccak(text=events[i % len(events)])

        tx = contract.functions.appendLifecycle(
            cid_bytes, event_type
        ).build_transaction({
            'from':     account.address,
            'nonce':    nonce_base + i,
            'gas':      100_000,
            'gasPrice': get_gas_price(w3),
            'chainId':  CHAIN_ID,
        })
        signed = account.sign_transaction(tx)

        t0      = time.time()
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        elapsed = time.time() - t0

        finalities.append(elapsed)
        gas_useds.append(receipt['gasUsed'])

    result = {
        'n':                n,
        'operation':        'appendLifecycle()',
        'mean_finality_s':  round(statistics.mean(finalities), 4),
        'min_finality_s':   round(min(finalities), 4),
        'max_finality_s':   round(max(finalities), 4),
        'p95_finality_s':   round(sorted(finalities)[int(n*0.95)], 4),
        'mean_gas':         round(statistics.mean(gas_useds)),
        'finality_list_s':  [round(f, 4) for f in finalities],
        'gas_list':         gas_useds,
    }
    print(f"  mean finality={result['mean_finality_s']}s  "
          f"mean gas={result['mean_gas']:,}")
    return result


# ═══════════════════════════════════════════════════════════════
# Benchmark 7: Multi-node 동기화 확인
# ═══════════════════════════════════════════════════════════════
def measure_node_sync(tx_hash_hex: str) -> Dict:
    """4개 노드가 동일한 블록/트랜잭션을 갖는지 확인"""
    print(f"\n[7] Multi-node 동기화 확인...")

    sync_results = {}
    for name, url in NODE_URLS.items():
        w3n = make_w3(url)
        if w3n is None:
            sync_results[name] = {'connected': False}
            continue
        try:
            block_num = w3n.eth.block_number
            peers     = int(w3n.net.peer_count)
            # 트랜잭션 조회
            tx_receipt = w3n.eth.get_transaction_receipt(tx_hash_hex)
            sync_results[name] = {
                'connected':   True,
                'block_number': block_num,
                'peers':        peers,
                'tx_found':     tx_receipt is not None,
                'tx_status':    tx_receipt['status'] if tx_receipt else None,
                'tx_block':     tx_receipt['blockNumber'] if tx_receipt else None,
            }
        except Exception as e:
            sync_results[name] = {'connected': True, 'error': str(e)}

    # 동기화 일치 여부
    blocks = [v['block_number'] for v in sync_results.values()
              if v.get('connected') and 'block_number' in v]
    tx_statuses = [v['tx_found'] for v in sync_results.values()
                   if v.get('connected')]

    result = {
        'node_results':     sync_results,
        'all_synced':       len(set(blocks)) <= 1 if blocks else False,
        'block_spread':     max(blocks) - min(blocks) if blocks else -1,
        'tx_found_all':     all(tx_statuses) if tx_statuses else False,
        'connected_nodes':  sum(1 for v in sync_results.values() if v.get('connected')),
    }
    for name, r in sync_results.items():
        status = '✓' if r.get('connected') else '✗'
        print(f"  {status} {name}: block={r.get('block_number','?')}  "
              f"peers={r.get('peers','?')}  "
              f"tx_found={r.get('tx_found','?')}")
    print(f"  All synced: {result['all_synced']}  "
          f"Block spread: {result['block_spread']}")
    return result


# ═══════════════════════════════════════════════════════════════
# 결과 저장 및 논문용 요약
# ═══════════════════════════════════════════════════════════════
def save_results(all_results: Dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON 상세 결과
    json_path = os.path.join(OUTPUT_DIR, f'qbft_benchmark_{ts}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # 논문용 요약 텍스트
    summary_path = os.path.join(OUTPUT_DIR, 'qbft_benchmark_summary.txt')
    lines = [
        '=' * 65,
        '  MDAF 4-Node QBFT Benchmark Results (Empirical)',
        f'  Timestamp: {ts}',
        f'  Contract:  {all_results.get("contract_address", "N/A")}',
        '=' * 65,
        '',
    ]

    if 'block_interval' in all_results:
        bi = all_results['block_interval']
        lines += [
            '[1] Block Production Interval',
            f'    mean  = {bi["mean_interval_s"]} s',
            f'    min   = {bi["min_interval_s"]} s',
            f'    max   = {bi["max_interval_s"]} s',
            f'    stdev = {bi["stdev_interval_s"]} s',
            '',
        ]

    if 'single_finality' in all_results:
        sf = all_results['single_finality']
        lines += [
            '[2] Single Transaction Finality  (ping)',
            f'    mean   = {sf["mean_finality_s"]} s',
            f'    median = {sf["median_finality_s"]} s',
            f'    p95    = {sf["p95_finality_s"]} s',
            f'    min    = {sf["min_finality_s"]} s',
            f'    max    = {sf["max_finality_s"]} s',
            f'    gas    = {sf["mean_gas"]:,}',
            '',
        ]

    if 'sequential_tps' in all_results:
        st = all_results['sequential_tps']
        lines += [
            '[3] Sequential TPS',
            f'    TPS (total)  = {st["TPS_total"]} tx/s',
            f'    TPS (send)   = {st["TPS_send"]} tx/s',
            f'    confirmed    = {st["n_confirmed"]}/{st["n_submitted"]}',
            f'    blocks used  = {st["blocks_used"]}',
            f'    tx/block     = {st["tx_per_block"]}',
            '',
        ]

    if 'burst_tps' in all_results:
        bt = all_results['burst_tps']
        lines += [
            '[4] Burst TPS',
            f'    TPS (burst send) = {bt["TPS_burst_send"]} tx/s',
            f'    TPS (confirmed)  = {bt["TPS_confirmed"]} tx/s',
            f'    workers          = {bt["workers"]}',
            f'    confirmed        = {bt["n_confirmed"]}/{bt["n_submitted"]}',
            '',
        ]

    if 'cert_issuance' in all_results:
        ci = all_results['cert_issuance']
        lines += [
            '[5] Certificate Issuance  (storeRecord)',
            f'    mean finality = {ci["mean_finality_s"]} s',
            f'    p95 finality  = {ci["p95_finality_s"]} s',
            f'    mean gas      = {ci["mean_gas"]:,}',
            f'    min/max gas   = {ci["min_gas"]:,} / {ci["max_gas"]:,}',
            '',
        ]

    if 'lifecycle_append' in all_results:
        la = all_results['lifecycle_append']
        lines += [
            '[6] Lifecycle Append  (appendLifecycle)',
            f'    mean finality = {la["mean_finality_s"]} s',
            f'    mean gas      = {la["mean_gas"]:,}',
            '',
        ]

    if 'node_sync' in all_results:
        ns = all_results['node_sync']
        lines += [
            '[7] Multi-node Synchronisation',
            f'    connected nodes = {ns["connected_nodes"]}/4',
            f'    all synced      = {ns["all_synced"]}',
            f'    block spread    = {ns["block_spread"]}',
            f'    tx found all    = {ns["tx_found_all"]}',
            '',
        ]

    lines += ['=' * 65]
    summary = '\n'.join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n  JSON   → {json_path}")
    print(f"  요약   → {summary_path}")
    return summary


# ═══════════════════════════════════════════════════════════════
# 메인 실행
# ═══════════════════════════════════════════════════════════════
def run_benchmarks(quick: bool = False) -> Dict:
    n_block   = 6  if quick else 10
    n_single  = 10 if quick else 30
    n_seq     = 20 if quick else 50
    n_burst   = 30 if quick else 100
    n_cert    = 5  if quick else 20
    n_life    = 5  if quick else 20

    print('=' * 65)
    print('  MDAF 4-Node QBFT Benchmark Suite')
    print(f'  Mode: {"Quick" if quick else "Full"}')
    print('=' * 65)

    # ── 연결 ────────────────────────────────────────────────────
    print('\n[0] 노드 연결 확인...')
    w3 = None
    for name, url in NODE_URLS.items():
        w3 = make_w3(url)
        if w3:
            print(f'  ✓ {name} ({url})')
            print(f'    chain_id={int(w3.eth.chain_id)}  '
                  f'block={w3.eth.block_number}  '
                  f'peers={int(w3.net.peer_count)}')
            break
    if not w3:
        print('\n  ✗ 연결 실패. Docker 네트워크를 먼저 기동하세요:')
        print('    cd blockchain/besu_4node && docker compose up -d')
        sys.exit(1)

    # 동기화 대기
    print('  노드 동기화 대기...')
    for _ in range(30):
        if int(w3.net.peer_count) >= 1:
            break
        time.sleep(2)
    print(f'  peers={int(w3.net.peer_count)} ✓')

    # ── 컨트랙트 컴파일 + 배포 ──────────────────────────────────
    print('\n[0] 컨트랙트 컴파일 및 배포...')
    abi, bytecode = compile_contract()
    account       = Account.from_key(DEPLOYER_KEY)
    contract, addr = deploy_contract(w3, abi, bytecode, account)

    all_results = {
        'timestamp':        datetime.datetime.now().isoformat(),
        'contract_address': addr,
        'chain_id':         int(w3.eth.chain_id),
        'node_url':         NODE_URLS['validator1'],
        'mode':             'quick' if quick else 'full',
    }

    # ── 벤치마크 실행 ──────────────────────────────────────────
    all_results['block_interval']  = measure_block_interval(w3, n_block)
    all_results['single_finality'] = measure_single_tx_finality(w3, contract, account, n_single)
    all_results['sequential_tps']  = measure_sequential_tps(w3, contract, account, n_seq)
    all_results['burst_tps']       = measure_burst_tps(w3, contract, account, n_burst)
    all_results['cert_issuance']   = measure_cert_issuance(w3, contract, account, n_cert)

    # lifecycle: 발급된 cert_id 재사용
    cert_ids = all_results['cert_issuance'].get('cert_ids', [])
    if cert_ids:
        all_results['lifecycle_append'] = measure_lifecycle_append(
            w3, contract, account, cert_ids, n_life)

    # 마지막 tx_hash로 노드 동기화 확인
    last_ping_nonce = (w3.eth.get_transaction_count(account.address) - 1)
    # ping 재실행 하나로 hash 확보
    ping_tx = contract.functions.ping().build_transaction({
        'from':     account.address,
        'nonce':    w3.eth.get_transaction_count(account.address),
        'gas':      60_000,
        'gasPrice': get_gas_price(w3),
        'chainId':  CHAIN_ID,
    })
    signed_ping  = account.sign_transaction(ping_tx)
    sync_tx_hash = w3.eth.send_raw_transaction(signed_ping.raw_transaction)
    w3.eth.wait_for_transaction_receipt(sync_tx_hash, timeout=30)
    time.sleep(2)  # 전파 대기

    all_results['node_sync'] = measure_node_sync(sync_tx_hash.hex())

    # ── 저장 및 출력 ────────────────────────────────────────────
    summary = save_results(all_results)

    print('\n' + '=' * 65)
    print('  BENCHMARK COMPLETE')
    print('=' * 65)
    print(summary)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDAF QBFT Benchmark')
    parser.add_argument('--quick', action='store_true',
                        help='소규모 빠른 테스트 (n 축소)')
    args = parser.parse_args()

    # mdaf 루트에서 실행되도록 path 설정
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    run_benchmarks(quick=args.quick)
