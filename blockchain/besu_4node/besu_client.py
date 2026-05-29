#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
besu_client.py
--------------
web3.py client for the 4-node Hyperledger Besu QBFT network.
Handles:
  - Node connection & health check
  - Smart contract compilation & deployment (CertificateRegistry, QualityGateway, LifeBankDirectory)
  - Transaction submission with real gas/finality measurement
  - QBFT network stats (TPS, block time, peer count)
"""

import time, json, os, hashlib
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
from eth_account.signers.local import LocalAccount

# ── Node RPC endpoints ────────────────────────────────────────
NODE_URLS = {
    'validator1': 'http://localhost:18545',
    'validator2': 'http://localhost:18547',
    'validator3': 'http://localhost:18548',
    'validator4': 'http://localhost:18549',
}

# Deployer account (funded in genesis alloc)
DEPLOYER_PRIVKEY = '0x8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63'
CHAIN_ID = 1337

# ── Minimal ABI + Bytecode for deployment ─────────────────────
# (Compiled from MDAPContracts.sol — simplified version for direct deployment)
CERTIFICATE_REGISTRY_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {"name": "batchHash",     "type": "bytes32"},
            {"name": "gradeHash",     "type": "bytes32"},
            {"name": "dataHash",      "type": "bytes32"},
            {"name": "issuerDid",     "type": "string"},
            {"name": "grade",         "type": "uint8"},
            {"name": "passage",       "type": "uint8"},
            {"name": "piREP",         "type": "bytes"},
        ],
        "name": "issueCertificate",
        "outputs": [{"name": "certId", "type": "bytes32"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "certId",    "type": "bytes32"},
            {"name": "eventType", "type": "bytes32"},
        ],
        "name": "updateLifecycleEvent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "certId", "type": "bytes32"}],
        "name": "getCertificate",
        "outputs": [
            {"name": "certId",        "type": "bytes32"},
            {"name": "batchHash",     "type": "bytes32"},
            {"name": "dataHash",      "type": "bytes32"},
            {"name": "issuerDid",     "type": "string"},
            {"name": "grade",         "type": "uint8"},
            {"name": "passageNumber", "type": "uint8"},
            {"name": "issueTimestamp","type": "uint256"},
            {"name": "status",        "type": "uint8"},
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "certId", "type": "bytes32"},
            {"indexed": False, "name": "grade",  "type": "uint8"},
        ],
        "name": "CertificateIssued",
        "type": "event"
    },
]

# Simplified Solidity bytecode (pre-compiled for direct deployment)
# Full contract: deploy via Hardhat/Truffle in production
CERTIFICATE_REGISTRY_BYTECODE = """
// Inline simplified bytecode placeholder
// In production: compile MDAPContracts.sol with solc 0.8.20
// and replace with actual bytecode
"""

# Simplified inline contract (Vyper-style for direct compilation)
SIMPLE_REGISTRY_SOURCE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CertificateRegistrySimple {
    struct Certificate {
        bytes32 certId;
        bytes32 batchHash;
        bytes32 dataHash;
        string  issuerDid;
        uint8   grade;
        uint8   passageNumber;
        uint256 issueTimestamp;
        uint8   status;
    }

    mapping(bytes32 => Certificate) public certificates;
    mapping(bytes32 => bytes32[])   public lifecycleHistory;
    address public owner;

    event CertificateIssued(bytes32 indexed certId, uint8 grade);
    event LifecycleEventRecorded(bytes32 indexed certId, bytes32 eventType);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    function issueCertificate(
        bytes32 batchHash,
        bytes32 gradeHash,
        bytes32 dataHash,
        string calldata issuerDid,
        uint8 grade,
        uint8 passage
    ) external onlyOwner returns (bytes32) {
        require(grade < 4, "Grade D: pre-policy filter");

        bytes32 certId = keccak256(
            abi.encodePacked(batchHash, block.timestamp, msg.sender)
        );

        certificates[certId] = Certificate({
            certId:         certId,
            batchHash:      batchHash,
            dataHash:       dataHash,
            issuerDid:      issuerDid,
            grade:          grade,
            passageNumber:  passage,
            issueTimestamp: block.timestamp,
            status:         0
        });

        emit CertificateIssued(certId, grade);
        return certId;
    }

    function updateLifecycleEvent(
        bytes32 certId,
        bytes32 eventType
    ) external onlyOwner {
        require(certificates[certId].issueTimestamp > 0, "Not found");
        bytes32 eventHash = keccak256(
            abi.encodePacked(certId, eventType, block.timestamp)
        );
        lifecycleHistory[certId].push(eventHash);
        emit LifecycleEventRecorded(certId, eventType);
    }

    function getCertificate(bytes32 certId)
        external view
        returns (Certificate memory)
    {
        return certificates[certId];
    }

    function getLifecycleCount(bytes32 certId)
        external view returns (uint256)
    {
        return lifecycleHistory[certId].length;
    }
}
'''


class BesuClient:
    """
    web3.py client for 4-node Besu QBFT network.
    Connects to validator1 (primary) by default.
    """

    def __init__(self, rpc_url: str = NODE_URLS['validator1'],
                 privkey: str = DEPLOYER_PRIVKEY):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
        # QBFT / PoA middleware
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.account: LocalAccount = Account.from_key(privkey)
        self.rpc_url = rpc_url
        self._contract_address = None
        self._contract = None

    # ── Connection ────────────────────────────────────────────
    def is_connected(self) -> bool:
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def wait_for_node(self, timeout: int = 60, poll: float = 2.0) -> bool:
        """Wait until node is ready and has peers."""
        print(f'[BesuClient] Waiting for node at {self.rpc_url}...')
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.w3.is_connected():
                    peers = int(self.w3.net.peer_count)
                    block = self.w3.eth.block_number
                    print(f'  Connected: block={block}, peers={peers}')
                    if peers >= 1:
                        return True
            except Exception as e:
                pass
            time.sleep(poll)
        print(f'[BesuClient] Timeout after {timeout}s')
        return False

    def get_network_info(self) -> dict:
        """Collect network status from all 4 nodes."""
        info = {}
        for name, url in NODE_URLS.items():
            try:
                w = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 5}))
                w.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                info[name] = {
                    'connected':  w.is_connected(),
                    'block':      w.eth.block_number if w.is_connected() else -1,
                    'peers':      int(w.net.peer_count) if w.is_connected() else -1,
                    'chain_id':   int(w.eth.chain_id) if w.is_connected() else -1,
                }
            except Exception as e:
                info[name] = {'connected': False, 'error': str(e)}
        return info

    # ── Contract deployment ───────────────────────────────────
    def deploy_contract(self) -> str:
        """
        Compile and deploy CertificateRegistrySimple.
        Returns deployed contract address.
        """
        try:
            from solcx import compile_source, install_solc, get_installed_solc_versions
        except ImportError:
            raise RuntimeError(
                "py-solc-x not installed. Run: pip install py-solc-x"
            )

        print('[BesuClient] Compiling CertificateRegistrySimple...')
        # 사용 가능한 버전 확인 후 최적 버전 선택
        installed = [str(v) for v in get_installed_solc_versions()]
        if '0.8.20' in installed:
            ver = '0.8.20'
        elif installed:
            ver = sorted(installed)[-1]   # 설치된 최신 버전
        else:
            print('  Installing solc 0.8.20...')
            install_solc('0.8.20')
            ver = '0.8.20'

        compiled = compile_source(
            SIMPLE_REGISTRY_SOURCE,
            output_values=['abi', 'bin'],
            solc_version=ver,
            evm_version='london',   # PUSH0 방지
        )
        contract_id, contract_interface = next(iter(compiled.items()))
        abi      = contract_interface['abi']
        bytecode = contract_interface['bin']

        print(f'  Bytecode size: {len(bytecode)//2} bytes')

        # Deploy
        contract = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        nonce    = self.w3.eth.get_transaction_count(self.account.address)
        try:
            gas_est = contract.constructor().estimate_gas({'from': self.account.address})
        except Exception:
            gas_est = 1_500_000
        # 노드에서 실제 gas price 조회
        try:
            gas_price = int(self.w3.eth.gas_price)
            if gas_price == 0:
                gas_price = 1_000_000_000
        except Exception:
            gas_price = 1_000_000_000
        print(f'  gas price: {gas_price:,} wei')

        tx = contract.constructor().build_transaction({
            'from':     self.account.address,
            'nonce':    nonce,
            'gas':      gas_est + 50000,
            'gasPrice': gas_price,
            'chainId':  CHAIN_ID,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f'  Deploy tx: {tx_hash.hex()}')

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        addr = receipt['contractAddress']
        print(f'  Contract deployed: {addr}')
        print(f'  Block: {receipt["blockNumber"]} | Gas used: {receipt["gasUsed"]}')

        self._contract_address = addr
        self._contract = self.w3.eth.contract(address=addr, abi=abi)
        self._abi = abi
        return addr

    def load_contract(self, address: str, abi: list = None):
        """Load existing deployed contract."""
        if abi is None:
            abi = CERTIFICATE_REGISTRY_ABI
        self._contract_address = address
        self._contract = self.w3.eth.contract(address=address, abi=abi)

    # ── Transaction methods ───────────────────────────────────
    def issue_certificate(self, batch_hash: bytes, grade_hash: bytes,
                           data_hash: bytes, issuer_did: str,
                           grade: int, passage: int) -> dict:
        """
        Call issueCertificate() on-chain.
        Returns: {cert_id, tx_hash, block, gas_used, finality_s}
        """
        assert self._contract, 'Contract not deployed/loaded'
        assert grade < 4, 'Grade D not allowed through QualityGateway'

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        try:
            gp = int(self.w3.eth.gas_price)
            gp = gp if gp > 0 else 1_000_000_000
        except Exception:
            gp = 1_000_000_000
        tx = self._contract.functions.issueCertificate(
            batch_hash, grade_hash, data_hash, issuer_did, grade, passage
        ).build_transaction({
            'from':     self.account.address,
            'nonce':    nonce,
            'gas':      300000,
            'gasPrice': gp,
            'chainId':  CHAIN_ID,
        })
        signed = self.account.sign_transaction(tx)

        t0 = time.time()
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        finality_s = time.time() - t0

        # Extract certId from event logs
        cert_id = None
        try:
            logs = self._contract.events.CertificateIssued().process_receipt(receipt)
            if logs:
                cert_id = logs[0]['args']['certId'].hex()
        except Exception:
            pass

        return {
            'cert_id':    cert_id,
            'tx_hash':    tx_hash.hex(),
            'block':      receipt['blockNumber'],
            'gas_used':   receipt['gasUsed'],
            'finality_s': round(finality_s, 3),
            'status':     'success' if receipt['status'] == 1 else 'reverted',
        }

    def update_lifecycle(self, cert_id_hex: str, event_type: str) -> dict:
        """Call updateLifecycleEvent() on-chain."""
        assert self._contract, 'Contract not deployed/loaded'

        cert_id_bytes = bytes.fromhex(cert_id_hex.replace('0x', '').zfill(64))
        event_bytes   = Web3.keccak(text=event_type)

        nonce = self.w3.eth.get_transaction_count(self.account.address)
        try:
            gp2 = int(self.w3.eth.gas_price)
            gp2 = gp2 if gp2 > 0 else 1_000_000_000
        except Exception:
            gp2 = 1_000_000_000
        tx = self._contract.functions.updateLifecycleEvent(
            cert_id_bytes, event_bytes
        ).build_transaction({
            'from':     self.account.address,
            'nonce':    nonce,
            'gas':      100000,
            'gasPrice': gp2,
            'chainId':  CHAIN_ID,
        })
        signed  = self.account.sign_transaction(tx)
        t0      = time.time()
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        return {
            'tx_hash':    tx_hash.hex(),
            'block':      receipt['blockNumber'],
            'gas_used':   receipt['gasUsed'],
            'finality_s': round(time.time() - t0, 3),
        }

    def query_certificate(self, cert_id_hex: str) -> dict:
        """Read certificate from chain (no gas)."""
        assert self._contract
        cert_id_bytes = bytes.fromhex(cert_id_hex.replace('0x', '').zfill(64))
        result = self._contract.functions.getCertificate(cert_id_bytes).call()
        return {
            'certId':         result[0].hex(),
            'batchHash':      result[1].hex(),
            'dataHash':       result[2].hex(),
            'issuerDid':      result[3],
            'grade':          result[4],
            'passageNumber':  result[5],
            'issueTimestamp': result[6],
            'status':         result[7],
        }

    # ── QBFT network stats ────────────────────────────────────
    def get_validators(self) -> list:
        """Get current QBFT validator list via eth_call."""
        try:
            result = self.w3.provider.make_request(
                'qbft_getValidatorsByBlockNumber', ['latest']
            )
            return result.get('result', [])
        except Exception:
            return []

    def measure_block_time(self, n_blocks: int = 10) -> dict:
        """Measure actual block production time over n_blocks."""
        print(f'[BesuClient] Measuring block time over {n_blocks} blocks...')
        start_block = self.w3.eth.block_number
        times = []

        for i in range(n_blocks):
            t0 = time.time()
            target = start_block + i + 1
            while self.w3.eth.block_number < target:
                time.sleep(0.1)
            elapsed = time.time() - t0
            times.append(elapsed)

        import statistics
        return {
            'n_blocks':     n_blocks,
            'mean_s':       round(statistics.mean(times), 3),
            'min_s':        round(min(times), 3),
            'max_s':        round(max(times), 3),
            'stdev_s':      round(statistics.stdev(times), 3) if len(times) > 1 else 0,
        }
