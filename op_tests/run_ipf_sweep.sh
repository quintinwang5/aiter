#!/bin/bash
# I$ prefetch sweep: swap each variant .co into the deployed slot, run the test,
# collect perf (us / TB/s) + correctness (nrms). Portable: derives paths from its
# own location so it works wherever the aiter repo is checked out (e.g. the node).
#
# Usage:
#   ./run_ipf_sweep.sh                      # default CFG below
#   CFG="-b 8 -kvh 1 -c 16384 -m 0" ./run_ipf_sweep.sh
#   ONLY="s2_52 baseline_noprefetch" ./run_ipf_sweep.sh   # subset
#
# NOTE: if you sweep the kvh=1 ctx=16384 shape, rebuild the metadata JIT first so
# nrms is correct (it hits the deep-split metadata fix):
#   rm -f $(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))")/jit/module_pa_metadata.so
# (perf timing is comparable even without it; only nrms needs the rebuilt module.)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SWEEP="$REPO/hsa/gfx1250/pa_decode_bf16/ipf_sweep"
DEPLOY="${DEPLOY:-$REPO/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_mtp0.co}"  # default mtp0 (matches default -m 0); for mtp=1 set DEPLOY=.../..._mtp1.co
TEST="$HERE/test_pa_decode_bf16_asm.py"

CFG="${CFG:--b 1 -kvh 1 -c 16384 -m 0}"
VARIANTS="${ONLY:-baseline_noprefetch s2_16 s2_32 s2_44 s2_52 big}"

# back up whatever is currently deployed, restore on exit
BAK="$(mktemp)"; cp -p "$DEPLOY" "$BAK"
restore() { cp -p "$BAK" "$DEPLOY"; rm -f "$BAK"; echo "[restored original deployed .co]"; }
trap restore EXIT

echo "REPO=$REPO"
echo "CFG=$CFG"
echo "variants: $VARIANTS"
echo "============================================================"
for v in $VARIANTS; do
  co="$SWEEP/pa_$v.co"
  if [ ! -f "$co" ]; then echo "[$v] MISSING $co"; continue; fi
  cp -p "$co" "$DEPLOY"
  md=$(md5sum "$DEPLOY" | cut -c1-8)
  echo "===== $v (md5=$md) ====="
  python3 "$TEST" $CFG 2>&1 | grep -iE 'us\b|TB/s|nrms|PASS|FAIL|err' | sed "s/^/[$v] /"
  echo
done
echo "============================================================"
echo "done. (original .co restored)"
