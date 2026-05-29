// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * MDAFBenchmark.sol
 * -----------------
 * Lightweight contract for QBFT performance benchmarking.
 * Three operation types to measure different gas/complexity levels:
 *   1. ping()             - minimal op (baseline)
 *   2. storeRecord()      - single storage write (simulates cert issuance)
 *   3. appendLifecycle()  - append to array (simulates lifecycle event)
 */
contract MDAFBenchmark {

    address public owner;
    uint256 public pingCount;

    struct Record {
        bytes32 batchHash;
        bytes32 dataHash;
        uint8   grade;
        uint8   passage;
        uint256 timestamp;
        bool    valid;
    }

    mapping(bytes32 => Record)   public records;
    mapping(bytes32 => bytes32[]) public lifecycleHistory;
    bytes32[] public allCertIds;

    event Ping(address indexed sender, uint256 count);
    event RecordStored(bytes32 indexed certId, uint8 grade, uint256 timestamp);
    event LifecycleAppended(bytes32 indexed certId, bytes32 eventHash, uint256 seqNo);

    constructor() {
        owner = msg.sender;
        pingCount = 0;
    }

    // -- Benchmark Op 1: Minimal transaction ------------------
    function ping() external {
        pingCount += 1;
        emit Ping(msg.sender, pingCount);
    }

    // -- Benchmark Op 2: Certificate issuance (storage write) -
    function storeRecord(
        bytes32 batchHash,
        bytes32 dataHash,
        uint8   grade,
        uint8   passage
    ) external returns (bytes32 certId) {
        certId = keccak256(abi.encodePacked(
            batchHash, block.timestamp, msg.sender, allCertIds.length
        ));
        records[certId] = Record({
            batchHash: batchHash,
            dataHash:  dataHash,
            grade:     grade,
            passage:   passage,
            timestamp: block.timestamp,
            valid:     true
        });
        allCertIds.push(certId);
        emit RecordStored(certId, grade, block.timestamp);
        return certId;
    }

    // -- Benchmark Op 3: Lifecycle event append ----------------
    function appendLifecycle(
        bytes32 certId,
        bytes32 eventType
    ) external {
        require(records[certId].valid, "Record not found");
        bytes32 eventHash = keccak256(abi.encodePacked(
            certId, eventType, block.timestamp,
            lifecycleHistory[certId].length
        ));
        lifecycleHistory[certId].push(eventHash);
        emit LifecycleAppended(
            certId, eventHash, lifecycleHistory[certId].length
        );
    }

    // -- View functions ----------------------------------------
    function getRecord(bytes32 certId)
        external view returns (Record memory)
    {
        return records[certId];
    }

    function getLifecycleCount(bytes32 certId)
        external view returns (uint256)
    {
        return lifecycleHistory[certId].length;
    }

    function getTotalRecords() external view returns (uint256) {
        return allCertIds.length;
    }
}
