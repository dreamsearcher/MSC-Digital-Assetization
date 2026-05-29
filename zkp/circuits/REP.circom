pragma circom 2.0.0;

include "/home/claude/mdaf/node_modules/circomlib/circuits/comparators.circom";
include "/home/claude/mdaf/node_modules/circomlib/circuits/gates.circom";

/*
 * REP — ReleaseEligibilityProof
 * Public:  mqs_threshold(70×100=7000), viability_threshold(80),
 *          passage_max(5), data_hash_lo, data_hash_hi
 * Private: mqs(×100 스케일), viability, passage, sterility,
 *          preimage_lo, preimage_hi
 * Prove:   mqs>=7000, viability>=80, passage<=5,
 *          sterility==1, hash==data_hash
 */
template REP() {
    signal input  mqs_threshold;
    signal input  viability_threshold;
    signal input  passage_max;
    signal input  data_hash_lo;
    signal input  data_hash_hi;

    signal input mqs;
    signal input viability;
    signal input passage;
    signal input sterility;
    signal input preimage_lo;
    signal input preimage_hi;

    signal output valid;

    component mqs_chk = GreaterEqThan(14);
    mqs_chk.in[0] <== mqs;
    mqs_chk.in[1] <== mqs_threshold * 100;

    component via_chk = GreaterEqThan(7);
    via_chk.in[0] <== viability;
    via_chk.in[1] <== viability_threshold;

    component pass_chk = GreaterEqThan(4);
    pass_chk.in[0] <== passage_max;
    pass_chk.in[1] <== passage;

    signal ster_sq;
    ster_sq <== sterility * (sterility - 1);
    ster_sq === 0;

    signal hash_lo_diff;
    hash_lo_diff <== preimage_lo - data_hash_lo;
    hash_lo_diff === 0;

    signal hash_hi_diff;
    hash_hi_diff <== preimage_hi - data_hash_hi;
    hash_hi_diff === 0;

    component and1 = AND(); and1.a <== mqs_chk.out;  and1.b <== via_chk.out;
    component and2 = AND(); and2.a <== and1.out;       and2.b <== pass_chk.out;
    component and3 = AND(); and3.a <== and2.out;       and3.b <== sterility;

    valid <== and3.out;
}

component main {public [mqs_threshold, viability_threshold, passage_max, data_hash_lo, data_hash_hi]} = REP();
