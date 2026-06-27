# Probe: find the GPU (device) kernel symbol vs the host launcher of the same name.
set pagination off
set print repeats 0
set breakpoint pending on
set confirm off
break _ZN5aiter36pa_decode_bf16_d64_page256_gqa8_tq16E
run
echo \n===== ALL symbols matching the kernel name (host + device) =====\n
info functions pa_decode_bf16_d64_page256_gqa8_tq16
echo \n===== loaded code objects / shared libs =====\n
info sharedlibrary
echo \n===== rocm agents/queues =====\n
info agents
info dispatches
echo \n===== continue once; expect to re-hit (host) or land on GPU =====\n
continue
echo \n===== after 2nd hit: where + threads =====\n
where
info threads
echo \n===== PROBE END =====\n
kill
quit
