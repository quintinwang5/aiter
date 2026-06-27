# ============================================================================
# rocgdb: dump PV-WMMA wave regs to localize tq16 ctx256 nondeterminism.
# Input deterministic (manual_seed(0)+fixed --scales) => regs that differ
# across runs = the nondeterministic source.
# Run: rocgdb -batch -x rocgdb_pv.gdb --args python op_tests/test_pa_decode_bf16_asm.py -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0
# ============================================================================
set pagination off
set print repeats 0
set breakpoint pending on          # kernel is a dynamically-loaded hsaco -> pending
set confirm off

# Break on the GPU kernel (resolves when the code object loads).
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run

# --- diagnostics: did we actually stop in the GPU kernel? ---
echo \n===== after run: breakpoint state =====\n
info breakpoints
echo \n===== are we on a GPU wave? =====\n
info threads
echo \n===== where =====\n
where

# If we ARE stopped in the kernel, jump to the first PV WMMA and dump.
# PV WMMA text offset 0x4828; kernel entry instr at 0x34 -> +0x47f4 from entry $pc.
# (If 'where' above shows we're NOT in the kernel, the break didn't resolve;
#  see notes in the chat for fixing the symbol.)
tbreak *($pc + 0x47f4)
continue

echo \n===== AT (near) FIRST PV WMMA =====\n
where
echo \n--- v_V (A) v122..v137 ---\n
p/x $v122
p/x $v124
p/x $v126
p/x $v128
p/x $v130
p/x $v132
p/x $v134
p/x $v136
echo \n--- v_SP (B/P) v2..v8 ---\n
p/x $v2
p/x $v4
p/x $v6
echo \n--- step, then D=v_R_iter v188..v195 ---\n
stepi
p/x $v188
p/x $v190
p/x $v192
p/x $v194
echo \n===== END DUMP =====\n
kill
quit
