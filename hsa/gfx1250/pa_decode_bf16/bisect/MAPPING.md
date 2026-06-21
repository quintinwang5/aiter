# tq16 +4GB memory-fault bisect

Each .co is the tq16 kernel with ONE global memory op neutralized (its global
address clamped to s_Q_addr, which is always mapped -> cannot fault, cannot
hang; memory counts still advance). Built from
PA_DECODE_D64_1TG_4W_PS.sp3.willa_fix.preload.rspill_2.tq16 with `var BISECT=N`.

| file        | BISECT | op neutralized                          |
|-------------|--------|-----------------------------------------|
| tq16_b0.co  | 0      | baseline (all ops live) -> should FAULT |
| tq16_b1.co  | 1      | Q load (global_load_b128 v_Q_addr)      |
| tq16_b2.co  | 2      | K TDM (tensor_load_to_lds s_K_g0)       |
| tq16_b3.co  | 3      | V TDM (tensor_load_to_lds s_V_g0)       |
| tq16_b4.co  | 4      | O store (global_store_async, direct)    |
| tq16_b5.co  | 5      | SplitLSE store (buffer_store)           |
| tq16_b6.co  | 6      | SplitO store (buffer_store)             |
| tq16_b7.co  | 7      | sink load (global_load_b32 v_SINK)      |

Deployed kernel name (what aiter loads): pa_decode_bf16_d64_page256_gqa8_tq16.co
To test one variant: copy tq16_bN.co over that name, then run the failing
small-batch (batch<32) test.

Auto-run all: edit DEPLOY/TEST in run_bisect.sh, then `bash run_bisect.sh`.
The first non-baseline B# that flips to NO FAULT identifies the culprit op.

Source switch: var BISECT in the .sp3 (line ~146). Per-op clamps are gated by
`if BISECT == N` at: Q load (Q_global_load_issue), compute_K_phys_addr (B2),
compute_V_phys_addr (B3), R_store_to_global async store (B4), store_partial_results
SplitLSE (B5) / SplitO (B6), sink load site (B7).
