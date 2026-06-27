#!/bin/bash
# One-shot: run the PV-WMMA rocgdb dump N times on the deterministic ctx256 input
# and diff v_V. Same input (q=k=v=0) => any v_V lane that differs across runs, or
# is nonzero, is the nondeterministic / uninitialized source at the PV WMMA.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

ROCGDB="${ROCGDB:-rocgdb}"
GDB="${GDB:-rocgdb_pv.gdb}"
REPS="${REPS:-3}"
TEST="op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0"
CO=hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co

# Deploy the kernel under debug (combined / real baseline).
if [ -x ./bisect_tq16.sh ]; then ./bisect_tq16.sh deploy real >/dev/null 2>&1; fi
echo "deployed co md5: $(md5sum $CO 2>/dev/null | cut -d' ' -f1)"
echo "gdb script: $GDB   test: $TEST"
echo

for n in $(seq 1 "$REPS"); do
  log="rocgdb_pv_$n.log"
  echo "=== run $n -> $log ==="
  $ROCGDB -batch -x "$GDB" --args python $TEST > "$log" 2>&1
  # show whether we reached the PV WMMA (pc) and the v_V values
  pc=$(grep -A1 'confirm pc' "$log" | grep -oE '0x[0-9a-f]+' | tail -1)
  echo "  pc at dump = ${pc:-<none>}  (entry was 0x...1b00; PV WMMA should be +0x4828)"
  # extract just the v_V dump block for diffing
  sed -n '/v_V (A operand)/,/END DUMP/p' "$log" | grep -E '^\$[0-9]+ =' > ".vv_$n"
  nz=$(grep -c -vE '\{(0x0, )*0x0\}' ".vv_$n")
  echo "  v_V dump lines: $(wc -l < ".vv_$n")  nonzero-lines: $nz  (0 nonzero = V is all-zero as expected)"
done

echo
echo "===== diff v_V across runs (same q=k=v=0 input) ====="
ok=1
for n in $(seq 2 "$REPS"); do
  if diff -q ".vv_1" ".vv_$n" >/dev/null; then
    echo "run1 == run$n : v_V identical"
  else
    echo "run1 != run$n : *** v_V NONDETERMINISTIC ***"; ok=0
    diff ".vv_1" ".vv_$n" | head -20
  fi
done
echo
if [ "$ok" = 1 ]; then
  echo "VERDICT: v_V deterministic across runs."
  echo "  -> if also all-zero: V is fine; nondeterminism is downstream (P/cvt/fold/reduce)."
  echo "  -> if nonzero-but-identical: V loaded wrong-but-deterministic (a layout bug, not a race)."
else
  echo "VERDICT: v_V differs run-to-run with identical input => uninitialized/racing V at PV WMMA = the bug source."
fi
