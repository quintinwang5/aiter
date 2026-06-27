#!/bin/bash
# Run the PV-WMMA dump N times; GROUP by workgroup; within each workgroup, check
# if v_V (input) and v_R_iter (output) are deterministic. Different workgroups
# legitimately differ (different data) -> only same-wg comparison is meaningful.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ROCGDB="${ROCGDB:-rocgdb}"; GDB="${GDB:-rocgdb_pv.gdb}"; REPS="${REPS:-6}"
TEST="op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0"
CO=hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co
[ -x ./bisect_tq16.sh ] && ./bisect_tq16.sh deploy real >/dev/null 2>&1
echo "co md5: $(md5sum $CO 2>/dev/null|cut -d' ' -f1)"

for n in $(seq 1 "$REPS"); do
  log="rocgdb_pv_$n.log"
  $ROCGDB -batch -x "$GDB" --args python $TEST > "$log" 2>&1
  # workgroup that this run's PV-WMMA breakpoint landed on:
  wg=$(grep -oE '\([0-9]+,[0-9]+,[0-9]+\)\[[0-9,]+\]' "$log" | tail -1)
  # v_V input (first 4 dumped) and v_R_iter output (8 dumped) as a signature:
  vv=$(sed -n '/v_V (A input)/,/PV OUTPUT/p' "$log" | grep -E '^\$[0-9]+ =' | md5sum | cut -d' ' -f1)
  out=$(sed -n '/PV OUTPUT v_R_iter/,/END DUMP/p' "$log" | grep -E '^\$[0-9]+ =' | md5sum | cut -d' ' -f1)
  echo "run$n  wg=${wg:-?}  vV=$vv  vR_iter=$out"
  echo "$wg" > ".wg_$n"; echo "$vv" > ".vvh_$n"; echo "$out" > ".outh_$n"
  # keep full output dump for later inspection
  sed -n '/PV OUTPUT v_R_iter/,/END DUMP/p' "$log" | grep -E '^\$[0-9]+ =' > ".out_$n"
done

echo
echo "===== group by workgroup; within a wg, vV and vR_iter MUST match if deterministic ====="
# crude grouping: for each pair with same wg, compare hashes
for a in $(seq 1 "$REPS"); do for b in $(seq $((a+1)) "$REPS"); do
  wa=$(cat ".wg_$a"); wb=$(cat ".wg_$b")
  [ "$wa" = "$wb" ] || continue
  vva=$(cat ".vvh_$a"); vvb=$(cat ".vvh_$b")
  oa=$(cat ".outh_$a"); ob=$(cat ".outh_$b")
  printf "wg=%s  run%s vs run%s:  vV %s   vR_iter %s\n" "$wa" "$a" "$b" \
     "$([ "$vva" = "$vvb" ] && echo SAME || echo DIFF)" \
     "$([ "$oa" = "$ob" ] && echo SAME || echo '*** DIFF (nondeterministic OUTPUT) ***')"
done; done
echo
echo "Interpretation:"
echo "  same-wg vV SAME + vR_iter SAME  -> PV is deterministic for that wg (bug elsewhere/other wg)"
echo "  same-wg vV SAME + vR_iter DIFF  -> PV WMMA output nondeterministic w/ same input => WMMA/P or output-reg race"
