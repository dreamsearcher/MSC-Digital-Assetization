#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_genesis.py
-------------------
Generate QBFT genesis.json for 4-node Hyperledger Besu network.
"""

import os, json

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')


def build_genesis(nodes: list) -> dict:
    """Build QBFT genesis.json from node list."""
    validator_addresses = [n['address'] for n in nodes]

    # QBFT extra data encoding
    # Format: RLP encode [vanity(32 bytes), validators[], votes[], round(0), seals[]]
    # Simplified: use known working format with validator list
    vanity = '0x' + '00' * 32

    # Encode validators in QBFT extraData format
    # Using Istanbul/QBFT compatible encoding
    extra_data = _encode_qbft_extra(validator_addresses)

    genesis = {
        "config": {
            "chainId": 1337,
            "berlinBlock": 0,
            "londonBlock": 0,
            "qbft": {
                "blockperiodseconds": 2,
                "epochlength": 30000,
                "requesttimeoutseconds": 4,
                "blockreward": "0",
                "mining": True
            }
        },
        "nonce": "0x0",
        "timestamp": "0x5b3d92d7",
        "gasLimit": "0x1fffffffffffff",
        "difficulty": "0x1",
        "mixHash": "0x63746963616c2062797a616e74696e65206661756c7420746f6c6572616e6365",
        "coinbase": "0x0000000000000000000000000000000000000000",
        "alloc": _build_alloc(nodes),
        "extraData": extra_data,
        "number": "0x0",
        "gasUsed": "0x0",
        "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
    }
    return genesis


def _encode_qbft_extra(validator_addresses: list) -> str:
    """
    Encode QBFT extraData using RLP.
    Format: RLP([32-byte vanity, [validators], [], 0, []])
    """
    try:
        from rlp import encode as rlp_encode
        import rlp

        class QBFTExtraData(rlp.Serializable):
            fields = [
                ('vanity',     rlp.sedes.Binary.fixed_length(32, allow_empty=True)),
                ('validators', rlp.sedes.CountableList(rlp.sedes.Binary.fixed_length(20))),
                ('votes',      rlp.sedes.CountableList(rlp.sedes.raw)),
                ('round',      rlp.sedes.big_endian_int),
                ('seals',      rlp.sedes.CountableList(rlp.sedes.raw)),
            ]

        vanity_bytes = b'\x00' * 32
        validator_bytes = [bytes.fromhex(a.replace('0x', '')) for a in validator_addresses]

        extra = QBFTExtraData(
            vanity=vanity_bytes,
            validators=validator_bytes,
            votes=[],
            round=0,
            seals=[],
        )
        encoded = rlp_encode(extra)
        return '0x' + encoded.hex()

    except ImportError:
        # Fallback: manual RLP for simple case
        return _manual_qbft_extra(validator_addresses)


def _manual_qbft_extra(validator_addresses: list) -> str:
    """Manual RLP encoding for QBFT extraData."""
    def rlp_encode_length(length, offset):
        if length < 56:
            return bytes([offset + length])
        elif length < 256**8:
            bl = length.bit_length()
            byte_len = (bl + 7) // 8
            return bytes([offset + 55 + byte_len]) + length.to_bytes(byte_len, 'big')
        raise ValueError(f'Too long: {length}')

    def rlp_item(data: bytes) -> bytes:
        if len(data) == 1 and data[0] < 0x80:
            return data
        return rlp_encode_length(len(data), 0x80) + data

    def rlp_list(items: list) -> bytes:
        payload = b''.join(items)
        return rlp_encode_length(len(payload), 0xC0) + payload

    vanity   = b'\x00' * 32
    val_items = [rlp_item(bytes.fromhex(a.replace('0x',''))) for a in validator_addresses]
    validators_list = rlp_list(val_items)
    empty_list = rlp_list([])
    round_enc = rlp_item(b'\x80')  # round = 0

    payload = (rlp_item(vanity) + validators_list +
               empty_list + empty_list + empty_list)
    extra = rlp_encode_length(len(payload), 0xC0) + payload
    return '0x' + extra.hex()


def _build_alloc(nodes: list) -> dict:
    """Pre-fund all validator accounts with 1000 ETH."""
    alloc = {}
    for node in nodes:
        addr = node['address'].replace('0x', '')
        alloc[addr] = {
            "balance": "0xDE0B6B3A7640000"  # 1 ETH in wei (×1000 for testing)
        }
    # Also fund a deployer account
    alloc["fe3b557e8fb62b89f4916b721be55ceb828dbd73"] = {
        "privateKey": "8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63",
        "comment": "Deployer account",
        "balance": "0xDE0B6B3A7640000000"
    }
    return alloc


def main():
    keys_path = os.path.join(CONFIG_DIR, 'node_keys.json')
    with open(keys_path) as f:
        nodes = json.load(f)

    genesis = build_genesis(nodes)
    out_path = os.path.join(CONFIG_DIR, 'genesis.json')
    with open(out_path, 'w') as f:
        json.dump(genesis, f, indent=2)
    print(f'genesis.json saved → {out_path}')
    print(f'ChainID: {genesis["config"]["chainId"]}')
    print(f'Validators: {[n["address"] for n in nodes]}')
    return genesis


if __name__ == '__main__':
    main()
