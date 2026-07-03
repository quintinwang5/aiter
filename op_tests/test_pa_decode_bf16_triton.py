# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton/Gluon counterpart of op_tests/test_pa_decode_bf16_asm.py.

Drives the Triton `unified_attention` paged-attention kernel on the SAME problem
grid the SP3 PA_DECODE_D64 asm test uses, so the two can be compared head-to-head
(对标).  Parameters are aligned with pa_decode_bf16_asm:

  * head_dim=64, page_size(block_size)=256, gqa=8.
  * FP8 Q **and** FP8 paged KV cache; bf16 output.
  * per-tensor scalar dequant scales for Q/K/V (q_descale/k_descale/v_descale).
  * softmax scale = 1/sqrt(head_dim); causal; optional GPT-OSS attention sink.

Style mirrors test_pa_decode_bf16_asm.py: an argparse grid over -b/-kvh/-c/-m
(+ --varlen/--scales/--sink/--context_lens) drives a torch reference vs the kernel
via aiter.test_common.checkAllclose, with per-config perf (us) reported and the
summary dumped to CSV.  The torch reference and paged-KV generation are reused from
op_tests/triton_tests/attention/test_unified_attention.py so this stays in lockstep
with the upstream triton test.
"""

import argparse
import itertools
import random
import sys
from typing import Optional, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

# Reuse the upstream triton test's paged-KV builder + torch reference so the
# numerics stay identical to test_unified_attention.py.
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

DEVICE_ARCH = arch_info.get_arch()
if DEVICE_ARCH not in ("gfx950", "gfx1250"):
    print(f"Skipping test_pa_decode_bf16_triton.py: requires gfx950/gfx1250, got {DEVICE_ARCH}")
    sys.exit(0)

torch.set_default_device("cuda")

# ---- aligned with test_pa_decode_bf16_asm.py kernel properties ----
PA_HEAD_DIM = 64
PA_PAGE_SIZE = 256
PA_GQA_RATIO = 8
PA_TILE_Q = 32  # mtp must be < PA_TILE_Q / gqa (= 4), matching the asm test


def rms_rel_err(ref, out):
    """RMS error normalized by peak magnitude (same metric as the asm test):
    nrms = sqrt(mean((ref-out)^2)) / max|.|."""
    a = ref.float()
    b = out.float()
    mag = max(a.abs().max().item(), b.abs().max().item(), 1e-9)
    return ((a - b).pow(2).mean().sqrt() / mag).item()


@perftest(num_rotate_args=1)
def run_unified_attention(
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
    shuffled_kv_cache,
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
        shuffled_kv_cache=shuffled_kv_cache,
    )
    return output


@benchmark()
def test_pa_decode_triton(
    batch: int,
    kv_head_num: int,
    ctx_len: int,
    mtp: int = 0,
    scales: Optional[Tuple[float, float, float]] = None,
    varlen: bool = False,
    use_sink: bool = False,
    context_lens: Optional[list] = None,
    shuffled_kv_cache: bool = True,
) -> dict:
    """Random FP8 paged inputs (arbitrary kv_len) vs the torch reference, timing the
    triton unified_attention kernel.

    Mirrors test_pa_decode_bf16_asm.test_pa_decode's config semantics:
      scales=None -> random per-tensor q/k/v dequant scales; else the given (q,k,v).
      mtp -> multi-token-predict layers (qlen = mtp+1); kernel requires mtp < 4.
      varlen -> random kv_len in [1, ctx_len] per sequence.
      context_lens -> explicit per-sequence kv lengths (overrides batch/ctx_len).
      use_sink -> per-Q-head GPT-OSS sink, added in both kernel and reference.
    """
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    assert mtp < PA_TILE_Q // gqa, f"kernel requires mtp < {PA_TILE_Q // gqa}, got {mtp}"
    qlen_with_mtp = mtp + 1
    num_query_heads = kv_head_num * gqa
    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    # ---- per-sequence (query_len, kv_len) pairs aligned with the asm grid ----
    if context_lens is not None:
        kv_lens_list = list(context_lens)
        batch = len(kv_lens_list)
    elif varlen:
        kv_lens_list = [max(int(random.uniform(1, ctx_len)), 1) for _ in range(batch)]
    else:
        kv_lens_list = [ctx_len] * batch
    seq_lens = [(qlen_with_mtp, kv) for kv in kv_lens_list]

    # ---- per-tensor q/k/v dequant scales (pa_asm convention: uniform(0.5, 2.0)) ----
    if scales is None:
        query_scale = round(random.uniform(0.5, 2.0), 4)
        key_scale = round(random.uniform(0.5, 2.0), 4)
        value_scale = round(random.uniform(0.5, 2.0), 4)
    else:
        query_scale, key_scale, value_scale = scales
    q_descale = torch.tensor([query_scale], dtype=dtypes.fp32, device=device)
    k_descale = torch.tensor([key_scale], dtype=dtypes.fp32, device=device)
    v_descale = torch.tensor([value_scale], dtype=dtypes.fp32, device=device)

    # ---- FP8 paged Q/KV inputs (reuse upstream generator) ----
    # num_blocks must cover the largest config: ceil(max_kv/page)*batch.
    max_blocks = ((max(kv_lens_list) + page_size - 1) // page_size) * batch
    num_blocks = max(64, max_blocks)
    (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _maybe_quant_query,
        _query_scales,
        _q_descale,
        _k_descale,
        _v_descale,
        _output_scale,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=page_size,
        head_size=head_dim,
        num_heads=(num_query_heads, kv_head_num),
        sliding_window=None,
        q_dtype=e4m3_dtype,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
        shuffled_kv_cache=shuffled_kv_cache,
        use_q_descale=False,  # supply our own pa_asm-style descales below
        use_kv_descale=False,
        use_out_scale=False,
        device=device,
    )

    sink = sinks if use_sink else None

    _, us = run_unified_attention(
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
        sink,
        shuffled_kv_cache,
    )
    torch.cuda.synchronize()

    ref = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=[qlen_with_mtp] * batch,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        sliding_window=None,
        soft_cap=None,
        sinks=sink,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=None,
    )

    err = checkAllclose(
        ref.float(),
        output.float(),
        atol=1.5e-1,
        rtol=1.5e-1,
        msg="[torch vs unified_attention][fp8]: us......",
    )
    nrms = rms_rel_err(ref, output)

    return {
        "max_kv": max(kv_lens_list),
        "mtp": mtp,
        "sink": use_sink,
        "qkv_scale": (query_scale, key_scale, value_scale),
        "shuffled": shuffled_kv_cache,
        "us": us,
        "err": err,
        "nrms": nrms,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of unified_attention (triton) pa-decode bench, aligned with test_pa_decode_bf16_asm.py",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    nargs="*",
    default=[1, 3, 8, 64],
    help="""Batch size.
    e.g. -b 1 3 8 64""",
)
parser.add_argument(
    "-kvh",
    "--kv_head_num",
    type=int,
    nargs="*",
    default=[1, 8],
    help="""Number of KV heads (q heads = kv_head_num * gqa(8)).
    e.g. -kvh 1 8""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    nargs="*",
    default=[7, 256, 1024, 4097, 16384],
    help="""Context length (arbitrary; multi-page when > 256).
    e.g. -c 256 4097""",
)
parser.add_argument(
    "-m",
    "--mtp",
    type=int,
    nargs="*",
    default=[0],
    help="""Multi-token-predict layers (qlen = mtp+1). Kernel requires mtp < 4.
    e.g. -m 0 1 2 3""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""Variable kv seqlens per batch (random in [1, ctx_len]). Default: False.""",
)
parser.add_argument(
    "--scales",
    type=float,
    nargs=3,
    default=None,
    metavar=("Q", "K", "V"),
    help="""Per-tensor q/k/v dequant scales by hand, e.g. --scales 0.5 2.0 1.5.
    Default: random scales per config.""",
)
parser.add_argument(
    "--sink",
    action="store_true",
    help="""Enable GPT-OSS attention sink: random per-Q-head sink logits passed to
    the kernel + matching sink term in the reference.""",
)
parser.add_argument(
    "--context_lens",
    type=int,
    nargs="*",
    default=None,
    help="""Explicit per-sequence context lengths for ONE shape. batch = number of
    values given; overrides -b/-c/--varlen. e.g. --context_lens 462 549 670 520 ...""",
)
parser.add_argument(
    "--no_shuffle",
    action="store_true",
    help="""Disable the pre-shuffled KV cache layout (default: shuffled, matching the
    optimized gfx1250 gluon path / the asm kernel's shuffled cache).""",
)
args = parser.parse_args()

# An explicit context_lens vector defines a single shape: batch = its length.
batch_sizes = [len(args.context_lens)] if args.context_lens else args.batch_size
ctx_lens = [max(args.context_lens)] if args.context_lens else args.ctx_len

df = []
for batch, kv_head_num, ctx_len, mtp in itertools.product(
    batch_sizes, args.kv_head_num, ctx_lens, args.mtp
):
    ret = test_pa_decode_triton(
        batch,
        kv_head_num,
        ctx_len,
        mtp,
        tuple(args.scales) if args.scales is not None else None,
        args.varlen,
        args.sink,
        args.context_lens,
        not args.no_shuffle,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("unified_attention (triton) pa-decode summary (markdown):\n%s", df_md)
df.to_csv("pa_decode_bf16_triton.csv")
