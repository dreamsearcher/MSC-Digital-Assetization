#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_besu.py — qbft_benchmark.py 래퍼 (기존 인터페이스 유지)
직접 실행 시: python qbft_benchmark.py [--quick]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from besu_client import NODE_URLS

def run_all_benchmarks(rpc_url=NODE_URLS['validator1']) -> dict:
    try:
        from qbft_benchmark import run_benchmarks
        return run_benchmarks(quick=False)
    except SystemExit:
        return {'error': 'Node not running'}

if __name__ == '__main__':
    import subprocess
    script = os.path.join(os.path.dirname(__file__), 'qbft_benchmark.py')
    subprocess.run([sys.executable, script] + sys.argv[1:])
