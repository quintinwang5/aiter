#!/bin/bash
# Bisect the tq16 +4GB memory fault by swapping in .co variants that each
# neutralize ONE global memory op (address clamped to s_Q_addr; cannot fault,
# cannot hang). Whichever variant STOPS faulting = the culprit op.
#
#   B0 = baseline (all ops live, should still fault)
#   B1 = Q load        B2 = K TDM      B3 = V TDM       B4 = O store
#   B5 = SplitLSE      B6 = SplitO     B7 = sink load
#
# Usage:
#   DEPLOY=/home/carhuang/qiwan/aiter/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co \
#   TEST='python op_tests/test_pa_decode_bf16_asm.py <small-batch args that fault>' \
#   bash run_bisect.sh
#
# IMPORTANT: TEST must be the SMALL-BATCH config that crashes (batch<32).

set -u
BISECT_DIR="$(cd "$(dirname "$0")" && pwd)/"
DEPLOY="${DEPLOY:?set DEPLOY to the .co path the test actually loads (see the aiter LoadKernel log line)}"
TEST="${TEST:?set TEST to the failing small-batch test command}"

[ -f "$DEPLOY" ] && cp -f "$DEPLOY" "${DEPLOY}.bak_bisect" 2>/dev/null

echo "DEPLOY = $DEPLOY"
echo "TEST   = $TEST"
echo "============================================================"
for N in 0 1 2 3 4 5 6 7 8; do
  cp -f "$BISECT_DIR/tq16_b$N.co" "$DEPLOY" || { echo "B$N: cp to DEPLOY failed (write perms?)"; exit 1; }
  # also mirror to the local install path in case that one is used
  cp -f "$BISECT_DIR/tq16_b$N.co" /local_vol1_nobackup/qiwan/mi400_aiter/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co 2>/dev/null
  LOG=/tmp/bisect_run_$N.log
  eval "$TEST" >"$LOG" 2>&1
  RC=$?
  if grep -qiE "Memory access fault|page not present|HSA_STATUS_ERROR_MEMORY|GPU coredump|Aborted" "$LOG"; then
    VERDICT="FAULT"
  elif [ $RC -ne 0 ]; then
    VERDICT="exit=$RC (no fault string; check $LOG)"
  else
    VERDICT="NO FAULT (ran to completion)"
  fi
  case $N in
    0) NAME="baseline (all live)";; 1) NAME="Q load";; 2) NAME="K TDM";; 3) NAME="V TDM";;
    4) NAME="O store";; 5) NAME="SplitLSE store";; 6) NAME="SplitO store";; 7) NAME="sink load";;
  esac
  printf "B%s  %-18s : %s\n" "$N" "$NAME" "$VERDICT"
done
echo "============================================================"
echo "The first NON-baseline B# that flips to 'NO FAULT' is the faulting op."
echo "(restore original with: cp ${DEPLOY}.bak_bisect $DEPLOY)"
