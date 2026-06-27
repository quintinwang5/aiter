#!/bin/bash
# Run the PV-WMMA rocgdb dump TWICE on the deterministic ctx256 input and diff.
# Same input (manual_seed(0) + --scales 1 1 1) => any VGPR that differs between
# the two runs is the nondeterministic source.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Debug the deployed kernel (combined / real baseline).
./bisect_tq16.sh deploy real >/dev/null 2>&1 || cp -f tq16_bisect_co/pa_decode_bf16_d64_page256_gqa8_tq16.co.fixA_real \
   hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co
echo "deployed: $(md5sum hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co | cut -d' ' -f1)"

ROCGDB="${ROCGDB:-rocgdb}"
TEST="op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0"

for n in 1 2 3; do
  echo "=== rocgdb run $n ==="
  $ROCGDB -batch -x rocgdb_pv.gdb --args python $TEST > rocgdb_run$n.log 2>&1
  echo "  wrote rocgdb_run$n.log"
done

echo
echo "===== diff run1 vs run2 (DUMP section only) ====="
sed -n '/AT FIRST PV WMMA/,/END DUMP/p' rocgdb_run1.log > .d1
sed -n '/AT FIRST PV WMMA/,/END DUMP/p' rocgdb_run2.log > .d2
diff .d1 .d2 && echo "run1==run2 (deterministic at this point)" || echo "^^^ these VGPRs are NONDETERMINISTIC ^^^"
echo "===== diff run1 vs run3 ====="
sed -n '/AT FIRST PV WMMA/,/END DUMP/p' rocgdb_run3.log > .d3
diff .d1 .d3 && echo "run1==run3" || echo "^^^ nondeterministic ^^^"
