#!/bin/bash
# ============================================================================
# BISECT the batch=1 page fault (tq16, post-fix source, rebuilt 2026-06-24).
# ============================================================================
# Each variant clamps ONE global-memory op's address to s_Q_addr (always mapped)
# so that op can neither fault nor hang (counts still advance). Whichever variant
# FLIPS from FAULT -> NO FAULT names the faulting op.
#
#   B0 = baseline (all ops live -> MUST still fault, else config is wrong)
#   B1=Q load  B2=K TDM  B3=V TDM  B4=O store  B5=SplitLSE  B6=SplitO
#   B7=sink load  B8=block-table lookup  B10=K+V TDM together
#
# IMPORTANT
#  * Run this ON the gfx1250 box (needs the GPU).
#  * Run from the mi400_aiter repo root (where op_tests/ lives).
#  * AITER_REBUILD=0 is forced below — otherwise aiter recompiles the .co from
#    source and silently overwrites the variant we just swapped in.
#  * The fault is in the SPLIT path: batch must be SMALL (1). Deep split only
#    happens with the real 256-TG grid, so this must run on silicon, not emu.
# ============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
BISECT_DIR="$HERE/bisect"

# The .co the test actually LoadKernel's (confirm against aiter's "hsaco:" log line).
# The user runs from carhuang's aiter checkout (per session history + real fault addrs).
DEPLOY="${DEPLOY:-/local_vol1_nobackup/qiwan/mi400_aiter/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co}"

# The failing batch=1 config. -c 1024 and 4096 both fault; 1024 is faster.
# NOTE: the kernel is sink-enabled, so the test MUST pass a non-null sink. Add
# --sink if your wrapper does not auto-fill it (else: "... sink must all be non-null").
# Override TEST to point at your aiter checkout / change the config.
AITER_ROOT="${AITER_ROOT:-/local_vol1_nobackup/qiwan/mi400_aiter}"
TEST="${TEST:-cd $AITER_ROOT && ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=0 \
  python3 op_tests/test_pa_decode_bf16_asm.py -b 1 -kvh 8 -c 1024 -m 0 --sink}"

echo "DEPLOY = $DEPLOY"
echo "TEST   = $TEST"
[ -f "$DEPLOY" ] && cp -f "$DEPLOY" "${DEPLOY}.bak_bisect"
echo "============================================================"
for N in 0 1 2 3 4 5 6 7 8 10; do
  CO="$BISECT_DIR/tq16_b$N.co"
  [ -f "$CO" ] || { echo "B$N: $CO missing — skip"; continue; }
  cp -f "$CO" "$DEPLOY" || { echo "B$N: cp to DEPLOY failed (perms?)"; exit 1; }
  LOG=/tmp/bisect_run_$N.log
  eval "$TEST" >"$LOG" 2>&1
  RC=$?
  if grep -qiE "Memory access fault|page not present|HSA_STATUS_ERROR_MEMORY|GPU core ?dump|Aborted|illegal" "$LOG"; then
    VERDICT="FAULT"
  elif [ $RC -ne 0 ]; then
    VERDICT="exit=$RC (no fault string; inspect $LOG)"
  else
    VERDICT="NO FAULT (ran to completion)"
  fi
  case $N in
    0) NM="baseline (all live)";; 1) NM="Q load";; 2) NM="K TDM";; 3) NM="V TDM";;
    4) NM="O store";; 5) NM="SplitLSE store";; 6) NM="SplitO store";; 7) NM="sink load";;
    8) NM="block-table lookup";; 10) NM="K+V TDM together";;
  esac
  printf "B%-2s %-20s : %s\n" "$N" "$NM" "$VERDICT"
done
echo "============================================================"
echo "B0 must FAULT. The first other B# that flips to NO FAULT = the faulting op."
echo "restore: cp ${DEPLOY}.bak_bisect $DEPLOY"
