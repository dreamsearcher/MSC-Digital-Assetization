#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
besu_adapter.py
---------------
Drop-in replacement for blockchain_simulator.py.
Uses real Besu 4-node QBFT network when available,
falls back to simulation if Docker not running.

Usage in run_full_pipeline.py:
    from blockchain.besu_4node.besu_adapter import MDAFBlockchainAdapter as MDAFBlockchainSystem
"""

import os, sys, json, time, hashlib
from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _is_besu_running(url: str = 'http://localhost:8545', timeout: int = 3) -> bool:
    """Check if Besu node is reachable."""
    try:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': timeout}))
        return w3.is_connected()
    except Exception:
        return False


class MDAFBlockchainAdapter:
    """
    Adapter that uses real Besu when available,
    simulation otherwise. Interface identical to MDAFBlockchainSystem.
    """

    def __init__(self, prefer_real: bool = True,
                 rpc_url: str = 'http://localhost:18545'):
        self._real = False
        self._client = None
        self._sim = None
        self._performance_log = []

        if prefer_real and _is_besu_running(rpc_url):
            try:
                from besu_client import BesuClient
                self._client = BesuClient(rpc_url=rpc_url)
                if self._client.is_connected():
                    print('[Adapter] ✓ Real Besu 4-node QBFT connected')
                    self._deploy_if_needed()   # 배포 성공 시 _real=True 설정
            except Exception as e:
                print(f'[Adapter] Besu connect failed: {e} → falling back to simulation')
                self._real = False
                self._client = None

        if not self._real:
            # Absolute import — works whether called from mdaf/ root or anywhere
            import importlib, sys
            # Ensure mdaf root is on path
            _mdaf_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            if _mdaf_root not in sys.path:
                sys.path.insert(0, _mdaf_root)
            from blockchain.blockchain_simulator import MDAFBlockchainSystem
            self._sim = MDAFBlockchainSystem()
            print('[Adapter] ⚠ Using simulation (Docker not running)')

    def _deploy_if_needed(self):
        """Deploy contract if not already deployed. Sets _real=True on success."""
        contract_cache = os.path.join(
            os.path.dirname(__file__), '..', '..', 'outputs', 'contract_address.txt'
        )
        # 캐시된 주소 먼저 시도
        if os.path.exists(contract_cache):
            with open(contract_cache) as f:
                addr = f.read().strip()
            try:
                self._client.load_contract(addr)
                print(f'[Adapter] Loaded existing contract: {addr}')
                self._real = True
                return
            except Exception:
                pass   # 캐시 무효 → 재배포
        # 신규 배포
        addr = self._client.deploy_contract()
        os.makedirs(os.path.dirname(contract_cache), exist_ok=True)
        with open(contract_cache, 'w') as f:
            f.write(addr)
        self._real = True   # ★ 배포 성공 후에만 real 모드 활성화
        print('[Adapter] ✓ Contract deployed and real mode activated')

    def process_batch(self, cert, data_hash: str) -> dict:
        """Issue certificate — real or simulated."""
        if self._real:
            return self._process_real(cert, data_hash)
        else:
            return self._sim.process_batch(cert, data_hash)

    def _process_real(self, cert, data_hash: str) -> dict:
        """Submit real on-chain transaction."""
        grade_map = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
        grade_int = grade_map.get(cert.mqs_grade, 4)

        if grade_int >= 4:
            return {'cert_id': cert.cert_id, 'grade': cert.mqs_grade,
                    'status': 'reverted', 'reason': 'Grade D pre-filtered',
                    'finality_s': 0, 'gas_used': 0}

        batch_hash  = Web3.keccak(text=cert.cert_id)
        grade_hash  = Web3.keccak(text=f'{cert.mqs_grade}_{cert.cert_id}')
        dh_bytes    = Web3.keccak(text=data_hash)

        t0 = time.time()
        result = self._client.issue_certificate(
            batch_hash, grade_hash, dh_bytes,
            cert.issuer_did, grade_int, cert.passage_number
        )
        elapsed = time.time() - t0

        self._performance_log.append({
            'op': 'issueCertificate',
            'grade': cert.mqs_grade,
            'finality_s': result['finality_s'],
            'gas_used':   result['gas_used'],
            'block':      result['block'],
        })

        return {
            'cert_id':    cert.cert_id,
            'grade':      cert.mqs_grade,
            'tx_hash':    result['tx_hash'],
            'block':      result['block'],
            'status':     result['status'],
            'gas_used':   result['gas_used'],
            'finality_s': result['finality_s'],
            'events':     [{'event': 'CertificateIssued'}],
        }

    def full_stats(self) -> dict:
        """Return performance stats."""
        if self._real:
            stats = {
                'mode':              'real_besu_4node',
                'connected':         self._client.is_connected(),
                'block_number':      self._client.w3.eth.block_number,
                'peer_count':        int(self._client.w3.net.peer_count),
                'chain_id':          int(self._client.w3.eth.chain_id),
                'transactions':      len(self._performance_log),
                'finality_s':        'measured',
            }
            if self._performance_log:
                fins = [r['finality_s'] for r in self._performance_log]
                import statistics as stat
                stats['mean_finality_s'] = round(stat.mean(fins), 3)
                stats['min_finality_s']  = round(min(fins), 3)
                stats['max_finality_s']  = round(max(fins), 3)
                stats['blocks']          = self._client.w3.eth.block_number
                stats['registered_certs']= len([r for r in self._performance_log
                                                 if r['op'] == 'issueCertificate'])
            return stats
        else:
            stats = self._sim.full_stats()
            stats['mode'] = 'simulation'
            return stats

    def get_performance_log(self) -> list:
        return self._performance_log

    # Proxy simulation-only attrs for compatibility
    def __getattr__(self, name):
        if self._sim and hasattr(self._sim, name):
            return getattr(self._sim, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
