# Stop at the FIRST PV WMMA (not kernel entry) of one GPU wave and dump V + output.
# Input q=k=v=0 (deterministic) => v_V should be 0; nonzero/varying = the bug.
set pagination off
set print repeats 0
set breakpoint pending on
set confirm off
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run
echo \n===== at kernel ENTRY (wave). pc below; PV WMMA = pc + 0x4828 =====\n
print/x $pc
# remove the entry breakpoint so 'continue' is NOT re-caught at entry by other waves
delete
# break at the first PV WMMA in THIS wave's code, then run to it
tbreak *($pc + 0x4828)
continue
echo \n===== should now be AT first PV WMMA: confirm pc =====\n
print/x $pc
where
echo \n--- v_V (A operand) v122..v137 : EXPECT 0 for q=k=v=0 ---\n
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
echo \n===== END DUMP =====\n
kill
quit
