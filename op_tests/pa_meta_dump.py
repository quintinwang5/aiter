"""Dump get_pa_metadata_v1() output for the PA-decode configs that diverge in the
V-mask test, so we can localize at the METADATA level (source-only, no binary
inspection):

  * kvh=1 ctx=1024  -> PASS (bitmatch)            <- known-good reference
  * kvh=1 ctx=7,256 -> out=inf (never written)    <- direct-to-O not executed?
  * kvh=1 ctx=4097  -> split_o written but RACES  <- the split-path race

For each config it prints the full metadata produced by build_pa_metadata()
(work_indptr, work_info reshaped to [-1,8], reduce_indptr/final_map/partial_map)
plus how many work items are direct-to-O vs split.  Run on silicon:

    python op_tests/pa_meta_dump.py

WorkInfo is printed as int rows; the 8 fields follow the kernel's PS ABI
(s68=batch_idx, s69=ploc/partial_idx, ..., s72=KV_start, s73=KV_end, ...).
"""
import torch
import aiter
from test_pa_decode_bf16_asm import build_pa_metadata, ceil_div

PAGE = 256
GQA = 8

# (batch, kv_head_num, ctx, mtp)
CONFIGS = [
    (1, 1, 7, 0),
    (1, 1, 256, 0),
    (1, 1, 1024, 0),   # known PASS
    (1, 1, 4097, 0),   # races
    (1, 8, 256, 0),
    (1, 8, 1024, 0),
    (1, 8, 4097, 0),
]


def build_one(batch, kv_head_num, ctx, mtp, device="cuda"):
    qlen = mtp + 1
    seq_lens_kv = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    actual_blocks = ceil_div(seq_lens_kv, PAGE)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_indices = torch.arange(
        int(kv_indptr[-1].item()), dtype=torch.int32, device=device
    )
    qo_indptr = torch.arange(
        0, (batch + 1) * qlen, qlen, dtype=torch.int32, device=device
    )
    return build_pa_metadata(
        batch, kv_head_num, GQA, qo_indptr, kv_indptr, seq_lens_kv,
        PAGE, qlen, device,
    ), kv_indptr, qo_indptr


def main():
    torch.set_printoptions(linewidth=200, threshold=10_000)
    for (b, kvh, ctx, mtp) in CONFIGS:
        (wi_ptr, winfo, rip, rfm, rpm, split_rows), kv_indptr, qo_indptr = build_one(
            b, kvh, ctx, mtp
        )
        w = winfo.view(-1, 8).cpu() if winfo.numel() else winfo.new_zeros((0, 8)).cpu()
        rip_c = rip.cpu().tolist()
        # direct-to-O vs split: group g is split iff rip[g+1] > rip[g]
        n_groups = max(0, len(rip_c) - 1)
        n_split = sum(1 for g in range(n_groups) if rip_c[g + 1] > rip_c[g])
        print("=" * 90)
        print(f"b={b} kvh={kvh} ctx={ctx} mtp={mtp}  pages/seq={ceil_div(torch.tensor([ctx]), PAGE).item()}")
        print(f"  kv_indptr        = {kv_indptr.cpu().tolist()}")
        print(f"  qo_indptr        = {qo_indptr.cpu().tolist()}")
        print(f"  split_rows       = {split_rows}")
        print(f"  n_work_items     = {w.shape[0]}")
        print(f"  work_indptr      = {wi_ptr.cpu().tolist()}")
        print(f"  reduce_indptr    = {rip_c}  (groups={n_groups}, split={n_split}, direct-O={n_groups - n_split})")
        print(f"  reduce_final_map = {rfm.cpu().view(-1, 2).tolist()}")
        print(f"  reduce_part_map  = {rpm.cpu().tolist()}")
        print(f"  work_info[-1,8]  (each row = one work item):")
        for i in range(w.shape[0]):
            print(f"    [{i:3d}] {w[i].tolist()}")


def dump_bin(batch, kvh, ctx, mtp, outdir):
    """Write aiter get_pa_metadata_v1 output in the emu's binary format so it can
    be substituted into the sim via META_DATA::load():

      work_indptr.bin : raw uint32[]            (num_tg+1 entries)
      work_info.bin   : raw WORK_INFO[]          (8x uint32 per work item)

    WORK_INFO field order matches the emu struct exactly:
      [batch_idx, partial_o_loc, qo_start, qo_end, kv_start, kv_end, kv_offset, q_head_range]
    -1 (direct-O ploc) has identical 0xFFFFFFFF bytes in int32/uint32.
    """
    import os
    import numpy as np

    (wi_ptr, winfo, _rip, _rfm, _rpm, _split_rows), _kvi, _qoi = build_one(
        batch, kvh, ctx, mtp
    )
    wi_ptr_np = wi_ptr.cpu().numpy().astype(np.uint32, copy=False)
    winfo_np = (
        winfo.view(-1, 8).cpu().numpy().astype(np.uint32, copy=False)
        if winfo.numel()
        else np.zeros((0, 8), dtype=np.uint32)
    )
    os.makedirs(outdir, exist_ok=True)
    wi_path = os.path.join(outdir, "work_indptr.bin")
    info_path = os.path.join(outdir, "work_info.bin")
    wi_ptr_np.tofile(wi_path)
    winfo_np.tofile(info_path)
    num_tg = wi_ptr_np.size - 1
    print(
        f"wrote {wi_path} ({wi_ptr_np.size} u32) and {info_path} "
        f"({winfo_np.shape[0]} work items x8 u32) for "
        f"b={batch} kvh={kvh} ctx={ctx} mtp={mtp}"
    )
    print(
        f">>> RUN THE EMU WITH  available_tgs={num_tg}  load_meta=1  "
        f"(grid_size_x must equal len(work_indptr)-1={num_tg}, else GPU reads work_indptr OOB)"
    )
    assert int(wi_ptr_np[-1]) == winfo_np.shape[0], (
        f"work_indptr.back()={int(wi_ptr_np[-1])} != num work items={winfo_np.shape[0]}; "
        "metadata inconsistent, emu generate_reduce_info would read work_info OOB"
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="store_true",
                    help="write work_indptr.bin/work_info.bin for one config instead of printing all")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--kvh", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4097)
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--out", default=".", help="output dir for the .bin files (the emu run dir)")
    args = ap.parse_args()

    if args.bin:
        dump_bin(args.batch, args.kvh, args.ctx, args.mtp, args.out)
    else:
        main()
