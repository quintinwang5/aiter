# Dump the FINAL normalized v_R (fp32, v66..v83) at the bf16-cvt (text 0xDF70),
# i.e. the value about to be stored. Group by workgroup; compare same-wg runs.
# Input q=k=v=0 fixed => same-wg differences across runs = the nondeterministic stage.
set pagination off
set print repeats 0
set breakpoint pending on
set confirm off
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run
delete
tbreak *($pc + 0xdf70)
continue
echo \n===== WORKGROUP =====\n
info threads
echo \n===== FINAL v_R (fp32, v66..v83) =====\n
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
