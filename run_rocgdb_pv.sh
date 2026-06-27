#!/bin/bash
# Dump FINAL v_R (post normalize) at many runs; group by workgroup; within a wg,
# the value MUST be identical if deterministic (input q=k=v=0 fixed).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ROCGDB="${ROCGDB:-rocgdb}"; GDB="${GDB:-rocgdb_pv.gdb}"; REPS="${REPS:-16}"
TEST="op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0"
CO=hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8_tq16.co
[ -x ./bisect_tq16.sh ] && ./bisect_tq16.sh deploy real >/dev/null 2>&1
echo "co md5: $(md5sum $CO 2>/dev/null|cut -d' ' -f1)"
rm -f rocgdb_pv_*.log
for n in $(seq 1 "$REPS"); do
  $ROCGDB -batch -x "$GDB" --args python $TEST > "rocgdb_pv_$n.log" 2>&1
  echo -n "."
done
echo
python3 - <<'PY'
import glob,re,hashlib,collections
runs={}
for f in sorted(glob.glob("rocgdb_pv_*.log"),key=lambda x:int(re.findall(r'\d+',x)[0])):
    t=open(f,errors='ignore').read()
    wg=re.findall(r'\((\d+,\d+,\d+)\)\[[0-9,]+\]',t); wg=wg[-1] if wg else "?"
    vr=re.findall(r'\$\d+ = \{[^}]*\}', t.split('FINAL v_R')[-1].split('END DUMP')[0]) if 'FINAL v_R' in t else []
    runs[f]=(wg, hashlib.md5("".join(vr).encode()).hexdigest()[:6], len(vr))
for f,(wg,h,n) in runs.items(): print(f"{f}: wg={wg:10s} finalVR={h} (#{n})")
print("\n--- same-wg groups (>=2 runs) ---")
bywg=collections.defaultdict(list)
for f,v in runs.items(): bywg[v[0]].append((f,v[1]))
hit=False
for wg,lst in sorted(bywg.items(),key=lambda x:len(x[1]),reverse=True):
    if len(lst)>=2:
        hit=True; hs={h for _,h in lst}
        verdict='SAME (deterministic)' if len(hs)==1 else '*** DIFF = NONDETERMINISTIC final v_R ***'
        print(f"wg={wg}: {len(lst)}x -> {verdict}")
if not hit: print("no same-wg repeat (raise REPS)")
PY
