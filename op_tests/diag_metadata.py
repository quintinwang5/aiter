#!/usr/bin/env python3
# Self-contained metadata-determinism diagnostic for batch=3, kv_head=1, ctx=16384.
# Runs get_pa_metadata_v1 TWICE on the SAME inputs and compares -> tells us whether
# the metadata DEVICE kernel itself is non-deterministic (the prime suspect now that
# 4097 is bit-exact and the compute-kernel empty-wave fix did nothing).
import sys, hashlib
import torch
import aiter


def ceil_div(a, b):
    return (a + b - 1) // b


def build_md(batch, kv_head_num, gqa, ctx_len, page_size, qlen, device):
    seq_lens_kv = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)
    actual_blocks = ceil_div(seq_lens_kv, page_size)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    qo_indptr = torch.arange(0, (batch + 1) * qlen, qlen, dtype=torch.int32, device=device)

    (
        (wmp_size, wmp_dtype),
        (wip_size, wip_dtype),
        (wi_size, wi_dtype),
        (ri_size, ri_dtype),
        (rfm_size, rfm_dtype),
        (rpm_size, rpm_dtype),
    ) = aiter.get_pa_metadata_info_v1(batch, kv_head_num)

    work_metadata_ptrs = torch.empty(wmp_size, dtype=wmp_dtype, device=device)
    work_indptr = torch.zeros(wip_size, dtype=wip_dtype, device=device)
    work_info = torch.zeros(wi_size, dtype=wi_dtype, device=device)
    reduce_indptr = torch.zeros(ri_size, dtype=ri_dtype, device=device)
    reduce_final_map = torch.zeros(rfm_size, dtype=rfm_dtype, device=device)
    reduce_partial_map = torch.zeros(rpm_size, dtype=rpm_dtype, device=device)

    aiter.get_pa_metadata_v1(
        qo_indptr, kv_indptr, seq_lens_kv,
        gqa, kv_head_num, True,
        work_metadata_ptrs, work_indptr, work_info,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=page_size, block_size=page_size,
        max_seqlen_qo=qlen, uni_seqlen_qo=qlen,
        fast_mode=True, max_split_per_batch=-1,
    )
    torch.cuda.synchronize()
    return work_indptr, work_info, reduce_indptr, reduce_final_map, reduce_partial_map, (wi_size, rpm_size)


def h(t):
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]


def main():
    dev = "cuda"
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU multiProcessorCount (num_cu) = {cu}")
    for ctx in (4097, 16384, 32768):
        print(f"\n==== batch=3 kvh=1 gqa=8 ctx={ctx} page=256 mtp=0 ====")
        runs = []
        for trial in range(3):
            wi, winfo, rip, rfm, rpm, (wi_size, rpm_size) = build_md(3, 1, 8, ctx, 256, 1, dev)
            ploc = winfo[:, 1]
            n_part = int((ploc >= 0).sum().item())
            max_ploc = int(ploc.max().item())
            n_work = int((winfo.abs().sum(dim=1) > 0).sum().item())
            runs.append((h(winfo), h(rip), h(rfm), h(rpm)))
            if trial == 0:
                print(f"  buffers: work_info_set={wi_size}  reduce_partial_map={rpm_size}")
                print(f"  n_work={n_work}  n_partials={n_part}  max_partial_qo_loc={max_ploc}")
                print(f"  split_rows(=rpm.numel*qlen)={rpm_size}  -> OOB if max_partial_qo_loc >= {rpm_size}: "
                      f"{'!!! OOB !!!' if max_ploc >= rpm_size else 'ok'}")
        same = all(r == runs[0] for r in runs)
        print(f"  metadata hashes (winfo,rip,rfm,rpm) x3: {runs}")
        print(f"  METADATA DETERMINISTIC: {same}" + ("" if same else "   <<< metadata kernel is NON-DETERMINISTIC"))


if __name__ == "__main__":
    main()
