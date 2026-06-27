# ============================================================================
# rocgdb script: dump PV-WMMA inputs/outputs of wave 0 to localize the
# tq16 ctx256 nondeterminism. Input is deterministic (manual_seed(0) + fixed
# --scales), so any VGPR that differs across two runs is the nondeterministic
# source.
#
# Usage (run TWICE, save each log, then diff):
#   cd /local_vol1_nobackup/qiwan/mi400_aiter
#   rocgdb -batch -x rocgdb_pv.gdb --args python op_tests/test_pa_decode_bf16_asm.py \
#          -b 64 -kvh 8 --scales 1.0 1.0 1.0 -c 256 -m 0 > run1.log 2>&1
#   ...repeat -> run2.log ; then: diff run1.log run2.log
# The "PASS/FAIL" run pair is the goal; diff shows which v-reg is nondeterministic.
# ============================================================================
set pagination off
set print repeats 0
set $K = "_ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E"

# 1) Stop when the tq16 kernel is first dispatched (device-code breakpoint).
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run

# 2) Compute the PV-WMMA address = kernel code base + first-PV-WMMA text offset.
#    The disasm's first instruction is at text 0x34; rocgdb stops at the kernel
#    entry ($pc). PV WMMA text offset = 0x4828, entry instr offset = 0x34, so
#    PV WMMA = $pc + (0x4828 - 0x34) = $pc + 0x47F4.
#    (If this misses, run `disassemble $pc,+0x5000` and find v_wmma_f32_16x16x128,
#     then `break *<that addr>` manually.)
set $pv = $pc + 0x47f4
tbreak *$pv
continue

# 3) We are now AT the first PV WMMA on some wave. Focus wave 0 if possible.
#    (rocgdb numbers GPU waves as threads; thread 1 is usually the first wave.
#     Adjust if your build differs; `info threads` lists them.)
echo \n===== AT FIRST PV WMMA (text 0x4828) =====\n
info threads

# 4) Dump the operands. For Q=K=V=0 these should be deterministic:
#    A = v_V  = v[122:137]  (expect 0)
#    B = v_SP = v[2:17]     (P, deterministic)
#    D = v_R_iter (output, written by THIS wmma) = v[188:195]
echo \n--- v_V (A, v122..v137) ---\n
p/x $v122
p/x $v123
p/x $v124
p/x $v125
p/x $v126
p/x $v127
p/x $v128
p/x $v129
p/x $v130
p/x $v131
p/x $v132
p/x $v133
p/x $v134
p/x $v135
p/x $v136
p/x $v137
echo \n--- v_SP (B/P, v2..v17) ---\n
p/x $v2
p/x $v3
p/x $v4
p/x $v5
echo \n--- step over the WMMA, then dump D=v_R_iter (v188..v195) ---\n
stepi
p/x $v188
p/x $v189
p/x $v190
p/x $v191
p/x $v192
p/x $v193
p/x $v194
p/x $v195

# 5) done
echo \n===== END DUMP =====\n
kill
quit
