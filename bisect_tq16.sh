#!/bin/bash
# ============================================================================
# tq16 single-real-wave (1-page-per-work) bisect helper
# ----------------------------------------------------------------------------
# Swaps the deployed _tq16.co between:
#   real  = fix-A real tq16 kernel        (pa_decode_bf16_d64_page256_gqa8_tq16.co.fixA_real)
#   diag  = tq32-compute / 16-row-output  (tq16_from_tq32_check.curso, built)
# then runs the PA decode asm test on the chosen configs.
#
# Usage:
#   ./bisect_tq16.sh deploy real          # install fix-A real kernel
#   ./bisect_tq16.sh deploy diag          # install diagnostic kernel
#   ./bisect_tq16.sh build diag           # (re)build diagnostic from .curso, install
#   ./bisect_tq16.sh build real           # (re)build fix-A real from .tq16.curso, install
#   ./bisect_tq16.sh test                 # run default bisect configs (ctx 256 16384 32768)
#   ./bisect_tq16.sh test -c 256 -m 0     # run custom args (passed through to the .py)
#   ./bisect_tq16.sh diag-test            # deploy diag + run default configs (one shot)
#   ./bisect_tq16.sh real-test            # deploy real + run default configs (one shot)
#   ./bisect_tq16.sh which                # show which .co is currently deployed (size/mtime/md5)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/local_vol1_nobackup/qiwan/sched2/pa_co_build"
SCHED2="/local_vol1_nobackup/qiwan/sched2"
AITER="/local_vol1_nobackup/qiwan/mi400_aiter"
CO_DST="$AITER/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co"

REAL_CO="$BUILD_DIR/pa_decode_bf16_d64_page256_gqa8_tq16.co.fixA_real"
DIAG_VARIANT="$SCHED2/PA_DECODE_D64_1TG_4W_PS.sp3.willa_fix.preload.rspill_2.tq16_from_tq32_check.curso"
REAL_VARIANT="$SCHED2/PA_DECODE_D64_1TG_4W_PS.sp3.willa_fix.preload.rspill_2.tq16.curso"

# Default configs: the two failing ctx (1-page and 128-page) + the passing 64-page.
DEF_ARGS=(-b 256 -kvh 8 -c 256 16384 32768 -m 0)

run_test() {
    cd "$AITER"
    echo "== test: deployed _tq16.co =="
    ls -la "$CO_DST"
    md5sum "$CO_DST" 2>/dev/null || true
    echo "== args: $* =="
    ENABLE_CK=0 ENABLE_FLYDSL=0 AITER_REBUILD=1 \
        python3 op_tests/test_pa_decode_bf16_asm.py "$@"
}

deploy() {
    case "$1" in
        real) cp -f "$REAL_CO" "$CO_DST"; echo "deployed REAL (fix-A) -> $CO_DST" ;;
        diag) # diag has no preserved .co; build it
              build diag ;;
        *) echo "deploy: arg must be 'real' or 'diag'"; exit 1 ;;
    esac
}

build() {
    case "$1" in
        diag) VARIANT="$DIAG_VARIANT" ;;
        real) VARIANT="$REAL_VARIANT" ;;
        *) echo "build: arg must be 'real' or 'diag'"; exit 1 ;;
    esac
    echo "== build+install $1 from $VARIANT =="
    VARIANT="$VARIANT" PRELOAD=30 STAGE=all bash "$BUILD_DIR/build.sh"
}

which_co() {
    echo "deployed: $CO_DST"
    ls -la "$CO_DST"; md5sum "$CO_DST"
    echo "--- reference md5s ---"
    [ -f "$REAL_CO" ] && md5sum "$REAL_CO" || echo "(no preserved real co)"
}

cmd="${1:-test}"; shift || true
case "$cmd" in
    deploy)    deploy "$@" ;;
    build)     build "$@" ;;
    which)     which_co ;;
    test)      if [ "$#" -gt 0 ]; then run_test "$@"; else run_test "${DEF_ARGS[@]}"; fi ;;
    diag-test) build diag; run_test "${DEF_ARGS[@]}" ;;
    real-test) deploy real; run_test "${DEF_ARGS[@]}" ;;
    *) echo "unknown cmd '$cmd' (deploy|build|which|test|diag-test|real-test)"; exit 1 ;;
esac
