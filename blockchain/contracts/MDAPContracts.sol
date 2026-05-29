// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ============================================================
// CertificateRegistry.sol
// MDAF MSC Digital Certificate Registry
// OpenZeppelin UUPS Upgradeable Pattern
// ============================================================

import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";

interface IZKPVerifier {
    function verifyREP(bytes calldata vk, bytes calldata publicInputs, bytes calldata proof) external view returns (bool);
    function verifyPQP(bytes calldata vk, bytes calldata publicInputs, bytes calldata proof) external view returns (bool);
}

contract CertificateRegistry is UUPSUpgradeable, AccessControlUpgradeable {

    bytes32 public constant LAB_ROLE      = keccak256("LAB_ROLE");
    bytes32 public constant BIOBANK_ROLE  = keccak256("BIOBANK_ROLE");
    bytes32 public constant AUDITOR_ROLE  = keccak256("AUDITOR_ROLE");

    enum CertStatus { Valid, Superseded, Revoked }
    enum Grade      { S, A, B, C, D }

    struct Certificate {
        bytes32   certId;
        bytes32   batchHash;
        bytes32   mqsGradeHash;
        bytes32   dataHash;
        string    issuerDid;
        bytes     releaseZkpProof;      // REP π
        bytes     premiumZkpProof;      // PQP π (0x00 if N/A)
        bytes32   zkpPolicyId;
        bytes32   lifecycleEventHash;
        uint8     passageNumber;
        bytes2    cellSourceCode;
        string    modelVersion;
        uint256   issueTimestamp;
        CertStatus status;
        Grade     grade;
    }

    mapping(bytes32 => Certificate) private _certificates;
    mapping(bytes32 => bytes32[])   private _lifecycleHistory;

    IZKPVerifier public zkpVerifier;

    event CertificateIssued(bytes32 indexed certId, Grade grade, bytes32 policyId);
    event LifecycleEventRecorded(bytes32 indexed certId, bytes32 eventType, bytes32 eventHash);
    event CertificateRevoked(bytes32 indexed certId, bytes32 reasonHash);
    event DigitalTwinMatchRecorded(bytes32 indexed certId, bytes32 matchScoreHash);
    event ZKPVerificationCompleted(bytes32 indexed certId, bool repValid, bool pqpValid);

    function initialize(address _zkpVerifier) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        zkpVerifier = IZKPVerifier(_zkpVerifier);
    }

    function issueCertificate(
        bytes32 batchHash, string calldata issuerDid, string calldata modelVer,
        bytes32 gradeHash, Grade grade, uint8 passage, bytes2 sourceCode,
        bytes32 dataHash, bytes calldata piREP, bytes calldata piPQP
    ) external onlyRole(LAB_ROLE) {
        // Grade D: rejection record only, no certificate
        require(grade != Grade.D, "Grade D: pre-policy filter, rejection only");
        require(grade != Grade.C, "Grade C: below REP policy threshold");

        // REP verification
        bytes memory repPublicInputs = abi.encode(dataHash, bytes32(0x01));
        bool repValid = zkpVerifier.verifyREP(bytes(""), repPublicInputs, piREP);
        require(repValid, "REP: invalid proof");

        // PQP for S/A grades
        bool pqpValid = false;
        if (grade == Grade.S || grade == Grade.A) {
            bytes memory pqpPublicInputs = abi.encode(dataHash, bytes32(0x02));
            pqpValid = zkpVerifier.verifyPQP(bytes(""), pqpPublicInputs, piPQP);
            require(pqpValid, "PQP: invalid proof for S/A grade");
        }

        bytes32 certId = keccak256(abi.encodePacked(batchHash, block.timestamp, msg.sender));
        bytes32 policyId = (pqpValid) ? bytes32(uint256(0x03)) : bytes32(uint256(0x01));

        _certificates[certId] = Certificate({
            certId:               certId,
            batchHash:            batchHash,
            mqsGradeHash:         gradeHash,
            dataHash:             dataHash,
            issuerDid:            issuerDid,
            releaseZkpProof:      piREP,
            premiumZkpProof:      piPQP,
            zkpPolicyId:          policyId,
            lifecycleEventHash:   keccak256(abi.encodePacked("CertificateIssued", certId)),
            passageNumber:        passage,
            cellSourceCode:       sourceCode,
            modelVersion:         modelVer,
            issueTimestamp:       block.timestamp,
            status:               CertStatus.Valid,
            grade:                grade
        });

        emit CertificateIssued(certId, grade, policyId);
        emit ZKPVerificationCompleted(certId, repValid, pqpValid);
    }

    function updateLifecycleEvent(
        bytes32 certId, bytes32 eventType
    ) external onlyRole(LAB_ROLE) {
        require(_certificates[certId].issueTimestamp > 0, "Certificate not found");
        require(_certificates[certId].status == CertStatus.Valid, "Certificate not valid");
        bytes32 eventHash = keccak256(abi.encodePacked(certId, eventType, block.timestamp));
        _certificates[certId].lifecycleEventHash = eventHash;
        _lifecycleHistory[certId].push(eventHash);
        emit LifecycleEventRecorded(certId, eventType, eventHash);
    }

    function revokeCertificate(bytes32 certId, bytes32 reasonHash) external onlyRole(AUDITOR_ROLE) {
        require(_certificates[certId].issueTimestamp > 0, "Certificate not found");
        _certificates[certId].status = CertStatus.Revoked;
        emit CertificateRevoked(certId, reasonHash);
    }

    function recordDigitalTwinMatch(
        bytes32 certId, bytes32 matchScoreHash
    ) external onlyRole(LAB_ROLE) {
        require(_certificates[certId].issueTimestamp > 0, "Certificate not found");
        emit DigitalTwinMatchRecorded(certId, matchScoreHash);
    }

    function getCertificate(bytes32 certId) external view returns (Certificate memory) {
        return _certificates[certId];
    }

    function _authorizeUpgrade(address newImpl) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}


// ============================================================
// QualityGateway.sol
// Two-tier ZKP gate: REP + PQP
// ============================================================
contract QualityGateway {

    IZKPVerifier public zkpVerifier;
    bytes32 public crsVkREP;
    bytes32 public crsVkPQP;

    event RejectionRecorded(bytes32 indexed batchHash, string reason, uint256 timestamp);

    constructor(address _zkpVerifier) {
        zkpVerifier = IZKPVerifier(_zkpVerifier);
    }

    function checkREP(
        bytes32 dataHash, bytes calldata piREP
    ) external view returns (bool) {
        bytes memory publicInputs = abi.encode(dataHash, bytes32(uint256(0x01)));
        return zkpVerifier.verifyREP(bytes(""), publicInputs, piREP);
    }

    function checkPQP(
        bytes32 dataHash, bytes calldata piPQP
    ) external view returns (bool) {
        bytes memory publicInputs = abi.encode(dataHash, bytes32(uint256(0x02)));
        return zkpVerifier.verifyPQP(bytes(""), publicInputs, piPQP);
    }

    function recordRejection(bytes32 batchHash, string calldata reason) external {
        emit RejectionRecorded(batchHash, reason, block.timestamp);
    }
}


// ============================================================
// LifeBankDirectory.sol
// DID Registration, VC Verification, RBAC
// ============================================================
contract LifeBankDirectory is AccessControlUpgradeable, UUPSUpgradeable {

    bytes32 public constant BIOBANK_ROLE  = keccak256("BIOBANK_ROLE");
    bytes32 public constant LAB_ROLE      = keccak256("LAB_ROLE");
    bytes32 public constant HOSPITAL_ROLE = keccak256("HOSPITAL_ROLE");
    bytes32 public constant AUDITOR_ROLE  = keccak256("AUDITOR_ROLE");
    bytes32 public constant REGULATOR_ROLE= keccak256("REGULATOR_ROLE");

    struct DIDRecord {
        string  did;
        bytes32 role;
        bytes32 vcHash;     // Hash of presented Verifiable Credential
        bool    active;
        uint256 registeredAt;
    }

    mapping(address => DIDRecord) private _didRegistry;
    mapping(bytes32 => bool)      private _revokedVCs;

    event DIDRegistered(address indexed account, string did, bytes32 role);
    event VCRevoked(bytes32 indexed vcHash);

    function initialize() public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    function registerDID(
        address account, string calldata did,
        bytes32 role, bytes32 vcHash
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(!_revokedVCs[vcHash], "VC has been revoked");
        _didRegistry[account] = DIDRecord({
            did: did, role: role, vcHash: vcHash,
            active: true, registeredAt: block.timestamp
        });
        _grantRole(role, account);
        emit DIDRegistered(account, did, role);
    }

    function revokeVC(bytes32 vcHash) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokedVCs[vcHash] = true;
        emit VCRevoked(vcHash);
    }

    function getDID(address account) external view returns (DIDRecord memory) {
        return _didRegistry[account];
    }

    function hasValidDID(address account) external view returns (bool) {
        DIDRecord memory rec = _didRegistry[account];
        return rec.active && !_revokedVCs[rec.vcHash];
    }

    function _authorizeUpgrade(address) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}
