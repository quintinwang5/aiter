#!/bin/bash
# Test multiple PA-decode .co variants in one session.
# Each variant is copied over the active tq16.co that the test loads, then the
# V-mask determinism test is run. Look for "PASS" / det_zero==0.
#
# If you deploy the .co elsewhere (carhuang tree), set CO_ACTIVE to THAT path.
set -u
CO_DIR=/local_vol1_nobackup/qiwan/mi400_aiter/hsa/gfx1250/pa_decode_bf16
CO_ACTIVE=$CO_DIR/pa_decode_bf16_d64_page256_gqa8_tq16.co
REPS=${PA_VMASK_REPS:-200}
CFG="-c 1024 4096"      # pure-split, no partial page; add 4097 to include partial

for v in baseline qkwait maxsync nuclear; do
  echo "==================================================================="
  echo "===== VARIANT: $v   (md5 $(md5sum $CO_DIR/pa_decode_bf16_d64_page256_gqa8_tq16.co.$v | cut -d' ' -f1)) ====="
  echo "==================================================================="
  cp -f $CO_DIR/pa_decode_bf16_d64_page256_gqa8_tq16.co.$v $CO_ACTIVE || { echo "copy failed"; exit 1; }
  # If you deploy elsewhere, also copy to that path here.
  PA_VMASK_REPS=$REPS python op_tests/test_pa_decode_bf16_asm.py --vmask $CFG 2>&1 \
     | grep -E "V-mask (PASS|RACE)|nondeterministic|det_zero" | head -20
  echo
done
echo "DONE. 'PASS' / det_zero=0 on a variant = that fix works."
