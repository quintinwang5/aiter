# Stop at the GPU page fault and dump the kv_indices-lookup operands so we can tell
# WHICH cause: garbage WorkInfo (huge kv_end), negative index, or a real clamp bug.
set pagination off
set confirm off
run
# (on fault rocgdb stops automatically at the faulting wave/instruction)
echo \n===== FAULTING PC / INSN =====\n
x/i $pc
echo \n===== address regs: s[100:101] = ptr_KVIndices(s8:9) + s31*4 =====\n
p/x $s8
p/x $s9
p/x $s100
p/x $s101
echo \n===== page / clamp / WorkInfo regs (s31=tdm_page s92=kv_end-1 s98=raw pg s72=kv_start s73=kv_end s34=wave_id) =====\n
p/d $s31
p/d $s92
p/d $s98
p/d $s72
p/d $s73
p/d $s34
echo \n===== workgroup =====\n
info threads
quit
