# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Same-input bench: SP3 asm `pa_decode_bf16_asm` vs Triton `unified_attention`.

Both kernels run on the SAME logical Q/K/V/block_tables/scales/sink, generated ONCE via the
upstream triton `generate_data`, each fed through its own memory layout:
  * Triton : standard [num_blocks, block_size, Hkv, D] -> shuffled cache (as generate_data).
  * ASM    : the SAME logical cache repacked into the swizzled fp8 layout
             K [pages, Hkv, D/16, page, 16], V [pages, Hkv, page/16, D, 16], plus
             kv_indices/kv_indptr + split metadata (build_pa_metadata) + host cpu_reduce_v1.

Per config it reports, for EACH kernel independently: us, effective HBM TB/s, err
(checkAllclose vs its own torch reference) and nrms (RMS-relative error vs its own ref).
No cross-kernel result comparison -- just the two side by side on identical inputs.

  python op_tests/pa_compare_asm_triton_sameinput.py -b 1 -kvh 8 -c 1024 -m 0
  python op_tests/pa_compare_asm_triton_sameinput.py -b 1 8 -kvh 8 -c 256 1024 4097 -m 0 --scales 1.0 1.0 1.0
"""

import argparse
import itertools
import random

import pandas as pd
import torch

import aiter  # noqa: F401
from aiter.test_common import checkAllclose, perftest
from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype

# Shared logical-input generator + triton torch reference (upstream triton test; importable,
# guarded by pytest -> no top-level argparse side effects).
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

# ASM path helpers + constants (test_pa_decode_bf16_asm has an `if __name__` guard -> safe import).
from op_tests.test_pa_decode_bf16_asm import (
    PA_GQA_RATIO,
    PA_HEAD_DIM,
    PA_PAGE_SIZE,
    PA_TILE_Q,
    build_pa_metadata,
    ceil_div,
    cpu_reduce_v1,
    ref_pa_decode,
    run_pa_stage,
)


def rms_rel(ref: torch.Tensor, out: torch.Tensor) -> float:
    """RMS(out-ref) / RMS(ref) in fp32."""
    ref = ref.float()
    out = out.float()
    denom = ref.pow(2).mean().sqrt().clamp_min(1e-12)
    return float((out - ref).pow(2).mean().sqrt() / denom)


@perftest(num_rotate_args=1)
def run_triton(
    query,
    key_cache,
    value_cache,
    output,
    cu_query_lens,
    max_query_len,
    kv_lens,
    max_kv_len,
    scale,
    window_size,
    block_tables,
    q_descale,
    k_descale,
    v_descale,
    sinks,
):
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_query_len,
        seqused_k=kv_lens,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        output_scale=None,
        shuffled_kv_cache=True,
    )
    return output


def compare_one(
    batch,
    kv_head_num,
    ctx_len,
    mtp=0,
    scales=None,
    varlen=False,
    use_sink=False,
    context_lens=None,
):
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    assert mtp < PA_TILE_Q // gqa, f"kernel requires mtp < {PA_TILE_Q // gqa}, got {mtp}"
    qlen = mtp + 1
    Hq = kv_head_num * gqa
    device = "cuda"

    # ---- per-sequence kv lengths (same grid as the two bench tests) ----
    if context_lens is not None:
        kv_lens_list = list(context_lens)
        batch = len(kv_lens_list)
    elif varlen:
        random.seed(0)
        kv_lens_list = [max(int(random.uniform(1, ctx_len)), 1) for _ in range(batch)]
    else:
        kv_lens_list = [ctx_len] * batch
    seq_lens = [(qlen, kv) for kv in kv_lens_list]

    max_blocks = ((max(kv_lens_list) + page_size - 1) // page_size) * batch
    num_blocks = max(64, max_blocks)

    # ---- SINGLE source of truth: generate logical Q/K/V/block_tables/sink ----
    (
        query,  # [sum(qlen), Hq, D] e4m3
        key_cache_orig,  # [num_blocks, page, Hkv, D] e4m3 (standard)
        value_cache_orig,  # [num_blocks, page, Hkv, D] e4m3 (standard)
        key_cache,  # shuffled (triton kernel layout)
        value_cache,  # shuffled
        sinks,  # [Hq]
        output,  # [sum(qlen), Hq, D] bf16
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,  # 1/sqrt(D)
        window_size,
        block_tables,  # [batch, max_blocks_per_seq]
        *_,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=page_size,
        head_size=head_dim,
        num_heads=(Hq, kv_head_num),
        sliding_window=None,
        q_dtype=e4m3_dtype,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
        shuffled_kv_cache=True,
        use_q_descale=False,  # we supply identical per-tensor scales to both below
        use_kv_descale=False,
        use_out_scale=False,
        device=device,
    )

    # ---- identical per-tensor q/k/v dequant scales for BOTH kernels ----
    if scales is None:
        query_scale = key_scale = value_scale = 1.0
    else:
        query_scale, key_scale, value_scale = scales
    q_desc = torch.tensor([query_scale], dtype=torch.float32, device=device)
    k_desc = torch.tensor([key_scale], dtype=torch.float32, device=device)
    v_desc = torch.tensor([value_scale], dtype=torch.float32, device=device)

    sink_tri = sinks if use_sink else None

    # ---- common byte count (identical problem -> identical traffic for both) ----
    # PA-decode is memory-bound: KV pages (fp8, K+V) dominate, plus Q in + O out.
    seq_lens_kv = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)
    actual_blocks = ceil_div(seq_lens_kv, page_size)
    kv_tokens = int(actual_blocks.sum().item()) * page_size
    kv_bytes = kv_tokens * kv_head_num * head_dim * key_cache_orig.element_size() * 2
    q_bytes = query.numel() * query.element_size()
    o_bytes = output.numel() * output.element_size()
    total_bytes = kv_bytes + q_bytes + o_bytes

    def tbps(us):
        return round((total_bytes / (us * 1e-6)) / 1e12, 2) if us > 0 else 0.0

    # ================= TRITON =================
    out_tri, us_tri = run_triton(
        query,
        key_cache,
        value_cache,
        output,
        cu_query_lens,
        max_query_len,
        kv_lens,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        q_desc,
        k_desc,
        v_desc,
        sink_tri,
    )
    torch.cuda.synchronize()
    ref_tri = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=[qlen] * batch,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        sliding_window=None,
        soft_cap=None,
        sinks=sink_tri,
        q_descale=q_desc,
        k_descale=k_desc,
        v_descale=v_desc,
        output_scale=None,
    )
    err_tri = checkAllclose(
        ref_tri.float(),
        out_tri.float(),
        atol=1.5e-1,
        rtol=1.5e-1,
        msg="[triton vs ref]:",
    )
    nrms_tri = rms_rel(ref_tri, out_tri)

    # ================= ASM (repack the SAME logical data) =================
    # Q: [batch*qlen, Hq, D] -> [batch, qlen, Hkv, gqa, D]  (qhead h = kv*gqa + g)
    Q_asm = query.reshape(batch, qlen, kv_head_num, gqa, head_dim).contiguous()
    # K std [blk, page, Hkv, D] -> asm [blk, Hkv, D//16, page, 16]
    #   K_asm[p,h,d16,t,i] == key_cache_orig[p, t, h, d16*16+i]
    K_asm = (
        key_cache_orig.permute(0, 2, 3, 1)  # [blk, Hkv, D, page]
        .reshape(num_blocks, kv_head_num, head_dim // 16, 16, page_size)
        .permute(0, 1, 2, 4, 3)  # [blk, Hkv, D//16, page, 16]
        .contiguous()
    )
    # V std [blk, page, Hkv, D] -> asm [blk, Hkv, page//16, D, 16]
    #   V_asm[p,h,t16,d,j] == value_cache_orig[p, t16*16+j, h, d]
    V_asm = (
        value_cache_orig.permute(0, 2, 1, 3)  # [blk, Hkv, page, D]
        .reshape(num_blocks, kv_head_num, page_size // 16, 16, head_dim)
        .permute(0, 1, 2, 4, 3)  # [blk, Hkv, page//16, D, 16]
        .contiguous()
    )

    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_indices = torch.cat(
        [block_tables[i, : int(actual_blocks[i].item())] for i in range(batch)]
    ).to(torch.int32)
    qo_indptr = torch.arange(
        0, (batch + 1) * qlen, qlen, dtype=torch.int32, device=device
    )

    (
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        split_rows,
    ) = build_pa_metadata(
        batch,
        kv_head_num,
        gqa,
        qo_indptr,
        kv_indptr,
        seq_lens_kv,
        page_size,
        qlen,
        device,
    )
    split_o = torch.zeros(
        (split_rows, 1, Hq, head_dim), dtype=torch.float32, device=device
    )
    split_lse = torch.full(
        (split_rows, 1, Hq, 1), float("-inf"), dtype=torch.float32, device=device
    )

    # Kernel always reads the sink slot -> pass a finite buffer (real sinks, or a
    # -1e30 no-op).  ref_pa_decode gets the same buffer (mirrors test_pa_decode_bf16_asm).
    if use_sink:
        sink_asm = sinks.to(torch.float32)
    else:
        sink_asm = torch.full((Hq,), -1.0e30, dtype=torch.float32, device=device)

    out_stage, us_asm = run_pa_stage(
        Q_asm,
        K_asm,
        V_asm,
        kv_indices,
        seq_lens_kv,
        scale,
        kv_indptr,
        gqa,
        mtp,
        q_desc,
        k_desc,
        v_desc,
        qo_indptr,
        work_indptr,
        work_info,
        split_o,
        split_lse,
        sink_asm,
    )
    torch.cuda.synchronize()
    out_asm = cpu_reduce_v1(
        out_stage,
        split_o,
        split_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qlen,
    )
    ref_asm = ref_pa_decode(
        Q_asm,
        K_asm,
        V_asm,
        kv_indices,
        kv_indptr,
        seq_lens_kv,
        gqa,
        query_scale,
        key_scale,
        value_scale,
        scale,
        sink_asm,
    )
    err_asm = checkAllclose(
        ref_asm.float(),
        out_asm.float(),
        atol=2e-2,
        rtol=2e-2,
        msg="[asm vs ref]:",
    )
    nrms_asm = rms_rel(ref_asm, out_asm)

    return {
        "batch": batch,
        "kvh": kv_head_num,
        "ctx": ctx_len if context_lens is None else max(kv_lens_list),
        "mtp": mtp,
        "sink": use_sink,
        "scales": (query_scale, key_scale, value_scale),
        "us_triton": round(us_tri, 2),
        "TBps_triton": tbps(us_tri),
        "err_triton": err_tri,
        "nrms_triton": nrms_tri,
        "us_asm": round(us_asm, 2),
        "TBps_asm": tbps(us_asm),
        "err_asm": err_asm,
        "nrms_asm": nrms_asm,
        "speedup(tri/asm)": round(us_tri / us_asm, 3) if us_asm > 0 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    p.add_argument("-b", "--batch_size", type=int, nargs="*", default=[1])
    p.add_argument("-kvh", "--kv_head_num", type=int, nargs="*", default=[8])
    p.add_argument("-c", "--ctx_len", type=int, nargs="*", default=[1024])
    p.add_argument("-m", "--mtp", type=int, nargs="*", default=[0])
    p.add_argument("--varlen", action="store_true")
    p.add_argument(
        "--scales",
        type=float,
        nargs=3,
        default=None,
        help="fixed q k v dequant scales for both kernels (default 1 1 1)",
    )
    p.add_argument("--sink", action="store_true")
    p.add_argument("--context_lens", type=int, nargs="*", default=None)
    a = p.parse_args()

    rows = []
    if a.context_lens is not None:
        for kvh, mtp in itertools.product(a.kv_head_num, a.mtp):
            rows.append(
                compare_one(
                    len(a.context_lens),
                    kvh,
                    max(a.context_lens),
                    mtp,
                    a.scales,
                    a.varlen,
                    a.sink,
                    a.context_lens,
                )
            )
    else:
        for b, kvh, c, mtp in itertools.product(
            a.batch_size, a.kv_head_num, a.ctx_len, a.mtp
        ):
            rows.append(compare_one(b, kvh, c, mtp, a.scales, a.varlen, a.sink, None))

    df = pd.DataFrame(rows)
    fmt = {c: "{:.2e}".format for c in ["nrms_triton", "nrms_asm"]}
    print(df.to_string(index=False, formatters=fmt))
    df.to_csv("pa_compare_asm_triton_sameinput.csv", index=False)
    try:
        print("\n" + df.to_markdown(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
