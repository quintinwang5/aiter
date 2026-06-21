# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-click perf comparison: SP3 asm PA-decode vs Triton unified_attention.

Runs both op_tests/test_pa_decode_bf16_asm.py and op_tests/test_pa_decode_bf16_triton.py
with the SAME shape grid (forwards -b/-kvh/-c/-m/--varlen/--scales/--sink/--context_lens
to each), then joins their CSVs on (batch, kv_head_num, ctx_len, mtp) and prints a
side-by-side us / speedup table.

  python op_tests/compare_pa_decode_asm_vs_triton.py -b 1 8 -kvh 8 -c 1024 4097 -m 0

speedup = us_asm / us_triton  (>1 -> triton faster; <1 -> asm faster).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASM = HERE / "test_pa_decode_bf16_asm.py"
TRITON = HERE / "test_pa_decode_bf16_triton.py"
ASM_CSV = REPO / "pa_decode_bf16_asm.csv"
TRITON_CSV = REPO / "pa_decode_bf16_triton.csv"

KEYS = ["batch", "kv_head_num", "ctx_len", "mtp"]


def common_args():
    """Parse only the shape-grid args both tests share, then rebuild the argv to
    forward verbatim (so defaults stay in sync with the two tests)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-b", "--batch_size", type=int, nargs="*")
    p.add_argument("-kvh", "--kv_head_num", type=int, nargs="*")
    p.add_argument("-c", "--ctx_len", type=int, nargs="*")
    p.add_argument("-m", "--mtp", type=int, nargs="*")
    p.add_argument("--varlen", action="store_true")
    p.add_argument("--scales", type=float, nargs=3, default=None)
    p.add_argument("--sink", action="store_true")
    p.add_argument("--context_lens", type=int, nargs="*", default=None)
    a = p.parse_args()

    fwd = []
    if a.batch_size:
        fwd += ["-b", *map(str, a.batch_size)]
    if a.kv_head_num:
        fwd += ["-kvh", *map(str, a.kv_head_num)]
    if a.ctx_len:
        fwd += ["-c", *map(str, a.ctx_len)]
    if a.mtp:
        fwd += ["-m", *map(str, a.mtp)]
    if a.varlen:
        fwd += ["--varlen"]
    if a.scales is not None:
        fwd += ["--scales", *map(str, a.scales)]
    if a.sink:
        fwd += ["--sink"]
    if a.context_lens is not None:
        fwd += ["--context_lens", *map(str, a.context_lens)]
    return fwd


def run(script, fwd):
    print(f"\n=== running {script.name} {' '.join(fwd)} ===", flush=True)
    subprocess.run([sys.executable, str(script), *fwd], cwd=str(REPO), check=True)


def main():
    fwd = common_args()
    run(ASM, fwd)
    run(TRITON, fwd)

    asm = pd.read_csv(ASM_CSV)
    tri = pd.read_csv(TRITON_CSV)
    keys = [k for k in KEYS if k in asm.columns and k in tri.columns]

    merged = asm.merge(tri, on=keys, suffixes=("_asm", "_triton"))
    merged["speedup(asm/triton)"] = merged["us_asm"] / merged["us_triton"]
    cols = keys + [
        c
        for c in [
            "max_kv_asm",
            "us_asm",
            "us_triton",
            "speedup(asm/triton)",
            "nrms_asm",
            "nrms_triton",
        ]
        if c in merged.columns
    ]
    table = merged[cols]
    out = REPO / "pa_decode_asm_vs_triton.csv"
    table.to_csv(out, index=False)
    print("\n=== asm vs triton (us, speedup) ===")
    print(table.to_markdown(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
