# Stop at first PV WMMA of one wave; print its WORKGROUP, dump V input AND the
# WMMA output v_R_iter (after stepping the WMMA). Compare ONLY same-workgroup runs.
set pagination off
set print repeats 0
set breakpoint pending on
set confirm off
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run
delete
tbreak *($pc + 0x4828)
continue
echo \n===== WORKGROUP (group same-wg runs before diffing) =====\n
info threads
echo \n===== PV-WMMA pc =====\n
print/x $pc
echo \n--- v_V (A input) v122,124,126,128 ---\n
p/x $v122
p/x $v124
p/x $v126
p/x $v128
echo \n--- step the WMMA, then PV OUTPUT v_R_iter v188..v195 ---\n
stepi
p/x $v188
p/x $v189
p/x $v190
p/x $v191
p/x $v192
p/x $v193
p/x $v194
p/x $v195
echo \n===== END DUMP =====\n
kill
quit
