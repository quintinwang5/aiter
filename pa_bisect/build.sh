#!/bin/bash
# Build the gfx1250 PA decode code object and install it into aiter.
#
# Two stages (may be different machines):
#   Stage 1 (CSIM, needs sp3 + libsp3.so with the MI400 backend):
#       sp3 compiles the .sp3 -> .hex, then sp3cvt embeds it into a ready-to-
#       assemble .s using the hand-written kernarg ABI template (kernarg.template.s,
#       which mirrors the 0x160 KernelArgs in asm_pa_decode_bf16.cu / pa_ps.cpp).
#   Stage 2 (ROCm, needs amdclang++ with gfx1250 support):
#       amdclang++ assembles the .s -> .co, which is copied into the aiter tree.
#
# Run with no args to do both stages on a host that has both toolchains, or set
# STAGE=1 / STAGE=2 to split across machines.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# --- configurable ----------------------------------------------------------
# .wsm.cornerfix = the willa-softmax (.wsm) baseline with the corner-case fix.
# This is the variant the deployed .co is built from; it carries the by-value
# softmax_scale kernarg (0x160) and the sink path, so it serves both sink and
# non-sink callers.
# PA_DECODE_D64_1TG_4W_PS.sp3.willa_fix.preload.rspill_2
VARIANT="${VARIANT:-/local_vol1_nobackup/qiwan/sched2/PA_DECODE_D64_1TG_4W_PS.sp3.willa_fix.preload.rspill_2.tq16}"
# Hardware kernarg preload: number of leading kernarg dwords the CP preloads into
# SGPRs (0 = off, legacy s_load ABI). Must match USE_KARG_PRELOAD in the .sp3 and
# PA_KARG_PRELOAD in pa_ps.cpp. 30 = the tight ABI's full preload region.
PRELOAD="${PRELOAD:-30}"
# Kernel symbol = Itanium mangling of aiter::<co-basename> (fmha_fwd_bf16.csv style).
# MUST match knl_name in pa_decode_bf16.csv.
#KERNEL_NAME="${KERNEL_NAME:-_ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E}"
#WORKBASE="${WORKBASE:-pa_decode_bf16_d64_page256_gqa8_tq16}"  # intermediate file stem + .co name
KERNEL_NAME="${KERNEL_NAME:-_ZN5aiter31pa_decode_bf16_d64_page256_gqa8E}"
WORKBASE="${WORKBASE:-pa_decode_bf16_d64_page256_gqa8}"  # intermediate file stem + .co name

# LDS = 327680 (NOT 163840): the kernel runs in WGP mode (two CUs combine LDS),
# so the group_segment_fixed_size is the combined 320 KB.  Using 163840 + the
# wrong descriptor produces a .co that runs but reads its LDS-staged K/V as zero
# (uniform softmax, all-zero output).
LDS_BYTES="${LDS_BYTES:-327680}"
VGPR="${VGPR:-1024}"
SGPR="${SGPR:-106}"
SP3="${SP3:-/home/tingchen/sp3}"
SP3_LIBDIR="${SP3_LIBDIR:-/home/tingchen}"
# Local preload-capable sp3cvt copy (poc_kl base + --preload/--kernarg-size).
# PRELOAD>0 emits .amdhsa_user_sgpr_kernarg_preload_length and sets kernarg_size
# to the packed 0x98 ABI; PRELOAD=0 falls back to the legacy non-preload .co.
# (The kernel .sp3 has USE_KARG_PRELOAD=1, so PRELOAD must be 30 to match it AND
# PA_KARG_PRELOAD=1 in the aiter host. A non-preload .co with this kernel faults:
# s2..s31 are never CP-loaded -> garbage pointers.)
SP3CVT="${SP3CVT:-/local_vol1_nobackup/qiwan/mi400_aiter/pa_bisect/sp3cvt_preload.py}"
# Packed preload kernarg ABI size in bytes (0x98). Used only when PRELOAD>0.
KARG_BYTES="${KARG_BYTES:-152}"
PA_PS_CPP="${PA_PS_CPP:-/local_vol1_nobackup/qiwan/sched2/pa_ps.cpp}"  # kernarg ABI source
PYTHON="${PYTHON:-python3.10}"                           # sp3cvt needs py>=3.8 (shlex.join)
# Local ROCm 7.2.4 LLVM (LLVM 22, has gfx1250) extracted from the rocm-llvm rpm;
# override with a system ROCm install if present.
AMDCLANG="${AMDCLANG:-/local_vol1_nobackup/qiwan/rocm_llvm/opt/rocm-7.2.4/lib/llvm/bin/amdclang++}"
AITER_CO="${AITER_CO:-/local_vol1_nobackup/qiwan/mi400_aiter/hsa/gfx1250/pa_decode_bf16/${WORKBASE}.co}"
STAGE="${STAGE:-all}"

WORK="$HERE/work"
mkdir -p "$WORK"
S_OUT="$WORK/${WORKBASE}.s"

stage1() {
    echo "== Stage 1: poc_kl sp3cvt compile + embed -> $S_OUT (symbol=$KERNEL_NAME) =="
    cp -f "$VARIANT" "$WORK/${WORKBASE}.sp3"
    # poc_kl sp3cvt writes its own <stem>.s from kernel_template.s (correct gfx1250
    # descriptor) and auto-discovers a *.cpp next to the .sp3 to patch the kernarg
    # ABI -> drop pa_ps.cpp in the work dir so the 22-arg 0x160 layout is parsed.
    cp -f "$PA_PS_CPP" "$WORK/pa_ps.cpp"
    rm -f "$S_OUT"   # force sp3cvt to start from its template, not a stale .s
    PRELOAD_ARGS=""
    if [ "${PRELOAD:-0}" -gt 0 ]; then
        PRELOAD_ARGS="--preload $PRELOAD --kernarg-size $KARG_BYTES"
        echo "   (HW kernarg preload ON: length=$PRELOAD dwords, kernarg_size=$KARG_BYTES)"
    else
        echo "   (HW kernarg preload OFF: legacy ABI)"
    fi
    ( cd "$WORK" && LD_LIBRARY_PATH="$SP3_LIBDIR" "$PYTHON" "$SP3CVT" \
        -i "$WORK/${WORKBASE}.sp3" -n "$KERNEL_NAME" \
        -lds "$LDS_BYTES" -v "$VGPR" -s "$SGPR" --sp3 "$SP3" $PRELOAD_ARGS )
    echo "   wrote $S_OUT"
}

stage2() {
    echo "== Stage 2: amdclang++ assemble -> .co =="
    [ -f "$S_OUT" ] || { echo "ERROR: $S_OUT not found; run STAGE=1 first"; exit 1; }
    "$AMDCLANG" -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
        "$S_OUT" -o "$WORK/${WORKBASE}.co"
    cp -f "$WORK/${WORKBASE}.co" "$AITER_CO"
    echo "   installed -> $AITER_CO"
}

case "$STAGE" in
    1) stage1 ;;
    2) stage2 ;;
    all) stage1; stage2 ;;
    *) echo "STAGE must be 1, 2, or all"; exit 1 ;;
esac
echo "Done."
