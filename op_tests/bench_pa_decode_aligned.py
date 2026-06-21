# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Aligned decode-only bench: SP3 asm PA-decode vs Triton unified_attention.

Both kernels are fed BYTE-IDENTICAL inputs and only the **decode** (mtp=0) GPU
kernel time is compared — an apples-to-apples view that the separate-script join
(compare_pa_decode_asm_vs_triton.py) cannot give, because:

  * Input alignment is exact on gfx1250: dtypes.fp8 == e4m3_dtype, and the asm
    paged KV layout is byte-for-byte the triton `shuffled_kv_cache` layout
        K[pages, kv_head, head_dim//16, page, 16]
        V[pages, kv_head, page//16, head_dim, 16]
    so ONE shuffled K/V tensor feeds both kernels; Q is the same data reshaped
    ([batch, q_heads, hd] <-> [batch, 1, kv_head, gqa, hd]); per-tensor q/k/v
    scales == q/k/v_descale; sink shares the scaled-logit (exp(sink)) convention.

  * Timed region is aligned: asm times the PA kernel only (its split-KV LSE merge
    runs on host, untimed); triton is called with skip_reduce=True so it returns
    after the attention kernel with NO GPU reduce.  Both = one attention kernel.

speedup(asm/triton) = asm_us / triton_us  (>1 -> triton faster).
"""

import argparse
import random
import sys
from typing import Optional, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import perftest
from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

# Import-safe now that the asm test guards its CLI under __main__.
from op_tests.test_pa_decode_bf16_asm import (
    make_sched2_metadata,
    cpu_reduce,
    rms_rel_err,
    ceil_div,
    PA_HEAD_DIM,
    PA_PAGE_SIZE,
    PA_GQA_RATIO,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    shuffle_kv_cache,
    ref_paged_attn,
)

DEVICE_ARCH = arch_info.get_arch()
if DEVICE_ARCH != "gfx1250":
    print(f"Skipping bench_pa_decode_aligned.py: requires gfx1250, got {DEVICE_ARCH}")
    sys.exit(0)

torch.set_default_device("cuda")

fp8 = dtypes.fp8
assert fp8 == e4m3_dtype, f"input alignment needs dtypes.fp8 == e4m3_dtype ({fp8} vs {e4m3_dtype})"


def build_aligned_inputs(
    batch, kv_head_num, ctx_len, varlen, scales, use_sink, context_lens, device="cuda"
):
    """One logical input set shared by both kernels (decode, qlen=1).

    Returns a dict with the shuffled K/V (fed verbatim to BOTH kernels), the per-
    kernel Q views, the unshuffled key/value caches for the torch reference, and
    all index / metadata tensors.
    """
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    q_head_num = kv_head_num * gqa
    torch.manual_seed(0)
    random.seed(0)

    # ---- per-sequence kv lengths (decode: query_len = 1) ----
    if context_lens is not None:
        kv_lens_list = list(context_lens)
        batch = len(kv_lens_list)
    elif varlen:
        kv_lens_list = [max(int(random.uniform(1, ctx_len)), 1) for _ in range(batch)]
    else:
        kv_lens_list = [ctx_len] * batch
    seq_lens_kv = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)

    # ---- per-tensor q/k/v dequant scales (pa_asm convention) ----
    if scales is None:
        query_scale = round(random.uniform(0.5, 2.0), 4)
        key_scale = round(random.uniform(0.5, 2.0), 4)
        value_scale = round(random.uniform(0.5, 2.0), 4)
    else:
        query_scale, key_scale, value_scale = scales
    q_descale = torch.tensor([query_scale], dtype=dtypes.fp32, device=device)
    k_descale = torch.tensor([key_scale], dtype=dtypes.fp32, device=device)
    v_descale = torch.tensor([value_scale], dtype=dtypes.fp32, device=device)
    softmax_scale = 1.0 / (head_dim**0.5)

    # ---- paged block tables -> kv_indices/kv_indptr (asm) + block_tables (triton) ----
    max_blocks_per_seq = ceil_div(int(seq_lens_kv.max().item()), page_size)
    num_blocks = max_blocks_per_seq * batch
    block_tables = (
        torch.randperm(num_blocks, device=device)
        .to(torch.int32)
        .reshape(batch, max_blocks_per_seq)
    )
    actual_blocks = ceil_div(seq_lens_kv, page_size)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_indices = torch.cat(
        [block_tables[i, : int(actual_blocks[i].item())] for i in range(batch)]
    ).to(torch.int32)
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q = qo_indptr  # decode: query_len = 1 per seq

    # ---- ONE logical Q / K / V, fed to both kernels ----
    # KV in triton's unshuffled [num_blocks, page, kv_head, head_dim] (== ref layout);
    # shuffle once -> the common physical layout both kernels read.
    key_cache_orig = (
        0.5 * torch.randn(num_blocks, page_size, kv_head_num, head_dim, device=device)
    ).to(fp8)
    value_cache_orig = (
        0.5 * torch.randn(num_blocks, page_size, kv_head_num, head_dim, device=device)
    ).to(fp8)
    K_shuf, V_shuf = shuffle_kv_cache(key_cache_orig, value_cache_orig)

    # Q: triton [num_tokens=batch, q_heads, head_dim]; asm [batch, 1, kv_head, gqa, head_dim].
    q_tri = (0.5 * torch.randn(batch, q_head_num, head_dim, device=device)).to(fp8)
    q_asm = q_tri.view(batch, 1, kv_head_num, gqa, head_dim)

    # ---- asm split-KV metadata + scratch (decode mtp=0 -> qlen_granularity=1) ----
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    (
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        split_rows,
    ) = make_sched2_metadata(
        batch, kv_head_num, gqa, qo_indptr, kv_indptr, seq_lens_kv, page_size, 1, num_cu, device
    )
    split_o = torch.zeros((split_rows, 1, q_head_num, head_dim), dtype=dtypes.fp32, device=device)
    split_lse = torch.full(
        (split_rows, 1, q_head_num, 1), float("-inf"), dtype=dtypes.fp32, device=device
    )

    # ---- sink (shared); finite no-op when disabled (asm always reads the slot) ----
    if use_sink:
        sink = (torch.randn(q_head_num, device=device) * 2.0).to(dtypes.fp32) * 0.125
    else:
        sink = torch.full((q_head_num,), -1.0e30, dtype=dtypes.fp32, device=device)

    return dict(
        batch=batch,
        gqa=gqa,
        kv_head_num=kv_head_num,
        q_head_num=q_head_num,
        head_dim=head_dim,
        page_size=page_size,
        kv_lens_list=kv_lens_list,
        seq_lens_kv=seq_lens_kv,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        block_tables=block_tables,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        qo_indptr=qo_indptr,
        cu_seqlens_q=cu_seqlens_q,
        max_kv=int(seq_lens_kv.max().item()),
        key_cache_orig=key_cache_orig,
        value_cache_orig=value_cache_orig,
        K_shuf=K_shuf,
        V_shuf=V_shuf,
        q_tri=q_tri,
        q_asm=q_asm,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        split_o=split_o,
        split_lse=split_lse,
        sink=sink,
        use_sink=use_sink,
    )


@perftest(num_rotate_args=1)
def _time_asm(inp, split_o, split_lse):
    # Decode PA kernel only (mtp=0); split-KV LSE merge happens on host (untimed).
    return aiter.pa_decode_bf16_asm(
        inp["q_asm"],
        inp["K_shuf"],
        inp["V_shuf"],
        inp["kv_indices"],
        inp["seq_lens_kv"],
        inp["softmax_scale"],
        inp["kv_indptr"],
        gqa=inp["gqa"],
        mtp=0,
        query_scale=inp["query_scale"],
        key_scale=inp["key_scale"],
        value_scale=inp["value_scale"],
        qo_indptr=inp["qo_indptr"],
        work_indptr=inp["work_indptr"],
        work_info=inp["work_info"],
        split_o=split_o,
        split_lse=split_lse,
        sink=inp["sink"],
    )


@perftest(num_rotate_args=1)
def _time_triton(inp, output):
    # Attention kernel only: skip_reduce=True returns before the GPU reduce, so the
    # timed region matches asm's PA-only decode stage.
    return unified_attention(
        q=inp["q_tri"],
        k=inp["K_shuf"],
        v=inp["V_shuf"],
        out=output,
        cu_seqlens_q=inp["cu_seqlens_q"],
        max_seqlen_q=1,
        seqused_k=inp["seq_lens_kv"],
        max_seqlen_k=inp["max_kv"],
        softmax_scale=inp["softmax_scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_tables"],
        softcap=0,
        q_descale=inp["q_descale"],
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        sinks=inp["sink"] if inp["use_sink"] else None,
        output_scale=None,
        shuffled_kv_cache=True,
        skip_reduce=True,
    )


def _correctness(inp):
    """Sanity: both kernels (with their merge/reduce ON) vs the SAME torch reference
    -> confirms the inputs really are equivalent.  Returns (asm_nrms, triton_nrms)."""
    device = "cuda"
    b, qh, hd = inp["batch"], inp["q_head_num"], inp["head_dim"]
    sink = inp["sink"] if inp["use_sink"] else None

    ref = ref_paged_attn(
        query=inp["q_tri"],
        key_cache=inp["key_cache_orig"],
        value_cache=inp["value_cache_orig"],
        query_lens=[1] * b,
        kv_lens=inp["kv_lens_list"],
        block_tables=inp["block_tables"],
        scale=inp["softmax_scale"],
        out_dtype=torch.bfloat16,
        sliding_window=None,
        soft_cap=None,
        sinks=sink,
        q_descale=inp["q_descale"],
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        output_scale=None,
    )  # [num_tokens=b, qh, hd]

    # ---- asm full path (PA + host merge) ----
    so = torch.zeros_like(inp["split_o"])
    sl = torch.full_like(inp["split_lse"], float("-inf"))
    out_asm = aiter.pa_decode_bf16_asm(
        inp["q_asm"], inp["K_shuf"], inp["V_shuf"], inp["kv_indices"], inp["seq_lens_kv"],
        inp["softmax_scale"], inp["kv_indptr"], gqa=inp["gqa"], mtp=0,
        query_scale=inp["query_scale"], key_scale=inp["key_scale"], value_scale=inp["value_scale"],
        qo_indptr=inp["qo_indptr"], work_indptr=inp["work_indptr"], work_info=inp["work_info"],
        split_o=so, split_lse=sl, sink=inp["sink"],
    )
    torch.cuda.synchronize()
    out_asm = cpu_reduce(
        out_asm, so, sl, inp["reduce_indptr"], inp["reduce_final_map"],
        inp["reduce_partial_map"], inp["gqa"],
    )
    asm_nrms = rms_rel_err(ref.view(b, qh, hd).float(), out_asm.view(b, qh, hd).float())

    # ---- triton full path (attention + GPU reduce) ----
    out_tri = torch.empty(b, qh, hd, dtype=torch.bfloat16, device=device)
    unified_attention(
        q=inp["q_tri"], k=inp["K_shuf"], v=inp["V_shuf"], out=out_tri,
        cu_seqlens_q=inp["cu_seqlens_q"], max_seqlen_q=1, seqused_k=inp["seq_lens_kv"],
        max_seqlen_k=inp["max_kv"], softmax_scale=inp["softmax_scale"], causal=True,
        window_size=(-1, -1), block_table=inp["block_tables"], softcap=0,
        q_descale=inp["q_descale"], k_descale=inp["k_descale"], v_descale=inp["v_descale"],
        sinks=sink, output_scale=None, shuffled_kv_cache=True,
    )
    torch.cuda.synchronize()
    tri_nrms = rms_rel_err(ref.float(), out_tri.float())
    return asm_nrms, tri_nrms


def run_one(batch, kv_head_num, ctx_len, varlen, scales, use_sink, context_lens):
    inp = build_aligned_inputs(
        batch, kv_head_num, ctx_len, varlen, scales, use_sink, context_lens
    )
    asm_nrms, tri_nrms = _correctness(inp)

    out_tri = torch.empty(
        inp["batch"], inp["q_head_num"], inp["head_dim"], dtype=torch.bfloat16, device="cuda"
    )
    _, asm_us = _time_asm(inp, inp["split_o"], inp["split_lse"])
    _, tri_us = _time_triton(inp, out_tri)
    torch.cuda.synchronize()

    return {
        "batch": inp["batch"],
        "kv_head_num": kv_head_num,
        "ctx_len": ctx_len,
        "max_kv": inp["max_kv"],
        "qkv_scale": (inp["query_scale"], inp["key_scale"], inp["value_scale"]),
        "sink": use_sink,
        "asm_us": asm_us,
        "triton_us": tri_us,
        "speedup(asm/triton)": asm_us / tri_us if tri_us else float("nan"),
        "asm_nrms": asm_nrms,
        "triton_nrms": tri_nrms,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="aligned decode-only bench: asm pa_decode vs triton unified_attention",
    )
    parser.add_argument("-b", "--batch_size", type=int, nargs="*", default=[1, 3, 8, 64])
    parser.add_argument("-kvh", "--kv_head_num", type=int, nargs="*", default=[1, 8])
    parser.add_argument(
        "-c", "--ctx_len", type=int, nargs="*", default=[7, 256, 1024, 4097, 16384]
    )
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--scales", type=float, nargs=3, default=None, metavar=("Q", "K", "V"))
    parser.add_argument("--sink", action="store_true")
    parser.add_argument("--context_lens", type=int, nargs="*", default=None)
    args = parser.parse_args()

    batch_sizes = [len(args.context_lens)] if args.context_lens else args.batch_size
    ctx_lens = [max(args.context_lens)] if args.context_lens else args.ctx_len
    scales = tuple(args.scales) if args.scales is not None else None

    import itertools

    df = []
    for batch, kv_head_num, ctx_len in itertools.product(
        batch_sizes, args.kv_head_num, ctx_lens
    ):
        df.append(
            run_one(batch, kv_head_num, ctx_len, args.varlen, scales, args.sink, args.context_lens)
        )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "aligned decode bench (asm vs triton, decode-only us):\n%s", df.to_markdown(index=False)
    )
    df.to_csv("pa_decode_aligned_decode.csv")


if __name__ == "__main__":
    main()
