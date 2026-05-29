#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_network.py
----------------
Complete network initialization:
  1. Generate 4 validator keys
  2. Build genesis.json with QBFT extraData
  3. Write static-nodes.json
  4. Write per-node private key files (data/nodeN/key)
  5. Validate all outputs
Run once before `docker compose up`.
"""

import os, sys, json, shutil

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
LOGS_DIR   = os.path.join(BASE_DIR, 'logs')

sys.path.insert(0, SCRIPT_DIR)
from generate_keys   import main as gen_keys
from generate_genesis import main as gen_genesis


def write_node_key_files(nodes: list):
    """Write private key to data/nodeN/key (Besu key file format)."""
    for node in nodes:
        node_dir = os.path.join(DATA_DIR, f'node{node["node_idx"]}')
        os.makedirs(node_dir, exist_ok=True)
        key_path = os.path.join(node_dir, 'key')
        # Besu expects raw hex private key without 0x prefix
        priv_hex = node['private_key'].replace('0x', '')
        with open(key_path, 'w') as f:
            f.write(priv_hex)
        print(f'  node{node["node_idx"]} key → {key_path}')


def write_static_nodes(nodes: list):
    """Write static-nodes.json for peer discovery."""
    enodes = [n['enode'] for n in nodes]
    path = os.path.join(CONFIG_DIR, 'static-nodes.json')
    with open(path, 'w') as f:
        json.dump(enodes, f, indent=2)
    print(f'static-nodes.json saved → {path}')
    for e in enodes:
        print(f'  {e}')


def write_besu_config(node_idx: int, ip: str):
    """Write per-node besu config.toml (optional, for overrides)."""
    config = f"""# Node {node_idx} config
data-path="/data"
genesis-file="/config/genesis.json"
rpc-http-enabled=true
rpc-http-host="0.0.0.0"
rpc-http-port=8545
rpc-http-api=["ETH","NET","QBFT","ADMIN","TXPOOL","WEB3"]
rpc-http-cors-origins=["*"]
host-allowlist=["*"]
p2p-host="{ip}"
p2p-port=30303
logging="INFO"
min-gas-price=0
"""
    path = os.path.join(CONFIG_DIR, f'config_node{node_idx}.toml')
    with open(path, 'w') as f:
        f.write(config)


def validate_setup():
    """Verify all required files exist."""
    required = [
        os.path.join(CONFIG_DIR, 'genesis.json'),
        os.path.join(CONFIG_DIR, 'node_keys.json'),
        os.path.join(CONFIG_DIR, 'static-nodes.json'),
    ] + [os.path.join(DATA_DIR, f'node{i}', 'key') for i in range(1, 5)]

    print('\nValidating setup...')
    all_ok = True
    for path in required:
        exists = os.path.exists(path)
        status = '✓' if exists else '✗'
        print(f'  {status} {path}')
        if not exists:
            all_ok = False
    return all_ok


def teardown():
    """Remove all generated data (for clean restart)."""
    for d in [DATA_DIR, LOGS_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f'Removed {d}')
    for fname in ['genesis.json', 'node_keys.json', 'static-nodes.json']:
        p = os.path.join(CONFIG_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f'Removed {p}')


def main(clean=False):
    if clean:
        teardown()

    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    print('='*55)
    print('  MDAF 4-Node Besu QBFT Network Setup')
    print('='*55)

    # Step 1: Generate keys
    print('\n[1] Generating validator keys...')
    nodes = gen_keys()

    # Step 2: Write node key files
    print('\n[2] Writing per-node key files...')
    write_node_key_files(nodes)

    # Step 3: Generate genesis
    print('\n[3] Generating genesis.json...')
    genesis = gen_genesis()

    # Step 4: Write static-nodes
    print('\n[4] Writing static-nodes.json...')
    write_static_nodes(nodes)

    # Step 5: Write per-node configs
    ips = ['172.20.0.11', '172.20.0.12', '172.20.0.13', '172.20.0.14']
    print('\n[5] Writing per-node besu configs...')
    for i, ip in enumerate(ips, 1):
        write_besu_config(i, ip)

    # Step 6: Validate
    ok = validate_setup()
    if ok:
        print('\n✓ Setup complete. Run:')
        print('  cd besu_4node && docker compose up -d')
    else:
        print('\n✗ Setup incomplete — check errors above.')
        sys.exit(1)

    return nodes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true',
                        help='Teardown existing data before setup')
    args = parser.parse_args()
    main(clean=args.clean)
