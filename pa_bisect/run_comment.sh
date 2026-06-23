#!/bin/bash
# ============================================================================
# Localize the batch=1 page fault by COMMENTING OUT each global-memory op
# (the op is physically removed from the kernel, not just address-clamped).
# Whichever variant FLIPS from FAULT -> NO FAULT names the faulting op.
#
#   baseline    = current source (all ops live -> MUST still fault)
#   cmt_q       = Q global_load            cmt_ktdm  = K TDM tensor_load
#   cmt_vtdm    = V TDM tensor_load        cmt_blocktable = block-table s_load
#                                          (replaced by s_phys_blk_idx=0, no cascade)
#   cmt_sink    = sink global_load         cmt_splitlse = SplitLSE buffer_store
#   cmt_splito  = SplitO buffer_store      cmt_ostore   = direct-O async store
#
# RUN ON the gfx1250 box, from carhuang's aiter checkout.
# AITER_REBUILD=0 forced (else aiter recompiles + overwrites the swapped .co).
# Kernel is sink-enabled -> test must pass a non-null sink (--sink), else the
# host guard "... sink must all be non-null" aborts before launch.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HERE/bisect_comment"

DEPLOY="${DEPLOY:-/local_vol1_nobackup/qiwan/mi400_aiter/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co}"
AITER_ROOT="${AITER_ROOT:-/local_vol1_nobackup/qiwan/mi400_aiter}"
TEST="${TEST:-cd $AITER_ROOT && ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=0 \
  python3 op_tests/test_pa_decode_bf16_asm.py -b 1 -kvh 8 -c 1024 -m 0 --sink}"

echo "DEPLOY = $DEPLOY"; echo "TEST = $TEST"
[ -f "$DEPLOY" ] && cp -f "$DEPLOY" "${DEPLOY}.bak_cmt"
echo "============================================================"
for V in baseline q ktdm vtdm blocktable sink splitlse splito ostore; do
  CO="$DIR/$([ "$V" = baseline ] && echo baseline || echo cmt_$V).co"
  [ -f "$CO" ] || { echo "$V: $CO missing"; continue; }
  cp -f "$CO" "$DEPLOY" || { echo "$V: cp to DEPLOY failed (perms?)"; exit 1; }
  LOG=/tmp/cmt_run_$V.log
  eval "$TEST" >"$LOG" 2>&1; RC=$?
  if grep -qiE "Memory access fault|page not present|HSA_STATUS_ERROR_MEMORY|core ?dump|Aborted|illegal" "$LOG"; then
    R="FAULT"
  elif [ $RC -ne 0 ]; then R="exit=$RC (no fault str; see $LOG)"; else R="NO FAULT (completed)"; fi
  printf "%-12s : %s\n" "$V" "$R"
done
echo "============================================================"
echo "baseline must FAULT. First cmt_* that flips to NO FAULT = the faulting op."
echo "restore: cp ${DEPLOY}.bak_cmt $DEPLOY"
