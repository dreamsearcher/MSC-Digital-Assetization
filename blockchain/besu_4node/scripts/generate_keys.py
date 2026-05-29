#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_keys.py
----------------
Generate 4 validator node keys for Hyperledger Besu QBFT network.
Outputs: node_keys.json, genesis_validators.txt
"""

import os, json, hashlib, secrets
from eth_account import Account
from eth_keys import keys as eth_keys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')

def safe_key_bytes(hex_str: str) -> bytes:
    """Ensure private key is exactly 32 bytes."""
    h = hex_str.replace('0x', '').zfill(64)
    return bytes.fromhex(h)


def generate_node_key(idx: int) -> dict:
    """Generate deterministic node key from index (for reproducibility)."""
    # Use secrets for initial entropy, then derive deterministically with seed
    seed = hashlib.sha256(f'mdaf_besu_node_{idx}_seed_2025'.encode()).hexdigest()
    priv_bytes = safe_key_bytes(seed)

    priv_key  = eth_keys.PrivateKey(priv_bytes)
    pub_key   = priv_key.public_key
    pub_hex   = pub_key.to_hex().replace('0x', '')

    # Besu expects 128-char uncompressed public key (without 04 prefix)
    assert len(pub_hex) == 128, f'pub_hex length {len(pub_hex)} != 128'

    address = Account.from_key(priv_bytes).address

    return {
        'node_idx':    idx,
        'private_key': '0x' + priv_bytes.hex(),
        'public_key':  '0x' + pub_hex,
        'address':     address,
        'enode':       f'enode://{pub_hex}@192.168.200.{10 + idx}:30303',
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nodes = []
    for i in range(1, 5):   # 4 validators
        node = generate_node_key(i)
        nodes.append(node)
        print(f'Node {i}: address={node["address"]}')

    # Save node_keys.json
    out_path = os.path.join(OUTPUT_DIR, 'node_keys.json')
    with open(out_path, 'w') as f:
        json.dump(nodes, f, indent=2)
    print(f'\nSaved → {out_path}')

    # Save validator addresses for genesis
    addrs_path = os.path.join(OUTPUT_DIR, 'validator_addresses.txt')
    with open(addrs_path, 'w') as f:
        for n in nodes:
            f.write(n['address'] + '\n')
    print(f'Saved → {addrs_path}')

    return nodes


if __name__ == '__main__':
    main()
