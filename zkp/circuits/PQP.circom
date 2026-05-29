pragma circom 2.0.0;

include "/home/claude/mdaf/node_modules/circomlib/circuits/comparators.circom";
include "/home/claude/mdaf/node_modules/circomlib/circuits/gates.circom";

/*
 * PQP — PremiumQualityProof
 * REP보다 엄격: mqs>=85, viability>=95, passage<=3 + 마커 3종
 */
template PQP() {
    signal input  mqs_threshold;
    signal input  viability_threshold;
    signal input  passage_max;
    signal input  data_hash_lo;
    signal input  data_hash_hi;

    signal input mqs;
    signal input viability;
    signal input passage;
    signal input sterility;
    signal input cd90_valid;
    signal input cd73_valid;
    signal input neg_marker_ok;
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

    signal ster_sq;   ster_sq   <== sterility     * (sterility     - 1); ster_sq   === 0;
    signal cd90_sq;   cd90_sq   <== cd90_valid    * (cd90_valid    - 1); cd90_sq   === 0;
    signal cd73_sq;   cd73_sq   <== cd73_valid    * (cd73_valid    - 1); cd73_sq   === 0;
    signal neg_sq;    neg_sq    <== neg_marker_ok * (neg_marker_ok - 1); neg_sq    === 0;

    signal hash_lo_diff; hash_lo_diff <== preimage_lo - data_hash_lo; hash_lo_diff === 0;
    signal hash_hi_diff; hash_hi_diff <== preimage_hi - data_hash_hi; hash_hi_diff === 0;

    component a1 = AND(); a1.a <== mqs_chk.out; a1.b <== via_chk.out;
    component a2 = AND(); a2.a <== a1.out;        a2.b <== pass_chk.out;
    component a3 = AND(); a3.a <== a2.out;        a3.b <== sterility;
    component a4 = AND(); a4.a <== a3.out;        a4.b <== cd90_valid;
    component a5 = AND(); a5.a <== a4.out;        a5.b <== cd73_valid;
    component a6 = AND(); a6.a <== a5.out;        a6.b <== neg_marker_ok;

    valid <== a6.out;
}

component main {public [mqs_threshold, viability_threshold, passage_max, data_hash_lo, data_hash_hi]} = PQP();
