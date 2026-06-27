# Catch the RE-ENTRY (2nd) work of a persistent wave at the final cvt (text 0xDF70)
# and dump its final v_R (v66..v83). Persistent re-entry (Bug B) is the suspect for
# the nondeterminism that the one-shot 1st-work probe missed.
set pagination off
set print repeats 0
set breakpoint pending on
set confirm off
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run
delete
python
import gdb, re
addr = int(gdb.parse_and_eval("$pc")) + 0xdf70
seen = {}
class ReentryBP(gdb.Breakpoint):
    def stop(self):
        try:
            info = gdb.execute("info threads", to_string=True)
        except Exception:
            return True
        cur = [l for l in info.splitlines() if l.lstrip().startswith('*')]
        if not cur:
            return False
        m = re.search(r'\((\d+,\d+,\d+)\)', cur[0])
        if not m:
            return False
        wg = m.group(1)
        seen[wg] = seen.get(wg, 0) + 1
        if seen[wg] >= 2:
            gdb.write("\n===== WG %s RE-ENTRY hit #%d =====\n" % (wg, seen[wg]))
            return True
        return False
ReentryBP("*0x%x" % addr)
end
continue
echo \n===== WORKGROUP (re-entry work) =====\n
info threads
echo \n===== FINAL v_R (fp32, v66..v83) re-entry work =====\n
p/x $v66
p/x $v67
p/x $v68
p/x $v69
p/x $v70
p/x $v71
p/x $v72
p/x $v73
p/x $v74
p/x $v75
p/x $v76
p/x $v77
p/x $v78
p/x $v79
p/x $v80
p/x $v81
p/x $v82
p/x $v83
echo \n===== END DUMP =====\n
kill
quit
