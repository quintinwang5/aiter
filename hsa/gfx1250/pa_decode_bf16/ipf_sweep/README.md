# PA-decode I$ prefetch sweep (2026-06-30)
baseline = khoist (343668ff, ENABLE_INST_PREFETCH=0). 热循环体 ~52KB (offset 432..52576) < 64K SQC.
site0@~124(prologue) site2@~432(loop-top) site1@~52576(reduce tail). all_waves=1. 全部 va_vdst=82 (确定性保持).

| .co | s0 | s2(热循环) | s1(tail) | s_prefetch | 说明 |
|---|---|---|---|---|---|
| pa_baseline_noprefetch.co | - | - | - | 0 | 参照(当前 khoist) |
| pa_s2_16.co | 8 | 16 | 8 | 8 | 极窄 |
| pa_s2_32.co | 8 | 32 | 8 | 12 | 窄(欠覆盖循环尾) |
| pa_s2_44.co | 8 | 44 | 8 | 15 | 中 |
| pa_s2_52.co | 8 | 52 | 8 | 17 | 全覆盖循环 |
| pa_big.co | 44 | 52 | 16 | 28 | 激进(site0也预热) |
