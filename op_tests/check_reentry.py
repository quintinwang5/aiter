"""Does batch64 kvh8 ctx256 have persistent re-entry (a wg processing >1 work)?
Run inside the test docker:  python3 op_tests/check_reentry.py
"""
import collections
import torch
from test_pa_decode_bf16_asm import build_pa_metadata, ceil_div

PAGE, GQA, dev = 256, 8, "cuda"

def report(b, kvh, ctx, mtp):
    qlen = mtp + 1
    seq = torch.full((b,), ctx, dtype=torch.int32, device=dev)
    ab = ceil_div(seq, PAGE)
    kvip = torch.zeros(b + 1, dtype=torch.int32, device=dev)
    kvip[1:] = torch.cumsum(ab, 0)
    kvidx = torch.arange(int(kvip[-1]), dtype=torch.int32, device=dev)
    qoip = torch.arange(0, (b + 1) * qlen, qlen, dtype=torch.int32, device=dev)
    out = build_pa_metadata(b, kvh, GQA, qoip, kvip, seq, PAGE, qlen, dev)
    wi_ptr, winfo = out[0], out[1]
    wip = wi_ptr.flatten().tolist()
    works = winfo.numel() // 8
    diffs = [wip[i + 1] - wip[i] for i in range(len(wip) - 1)]
    dist = dict(collections.Counter(diffs))
    mpc = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n=== b={b} kvh={kvh} ctx={ctx} mtp={mtp} ===")
    print(f"  total works            : {works}")
    print(f"  grid wgs (len indptr-1): {len(diffs)}")
    print(f"  works-per-wg dist      : {dist}")
    print(f"  MAX works on one wg    : {max(diffs) if diffs else 0}")
    print(f"  multiProcessorCount    : {mpc}")
    print(f"  RE-ENTRY?              : {'YES (some wg does >1 work)' if (diffs and max(diffs)>1) else 'NO (1 work/wg)'}")

if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    report(64, 8, 256, 0)
    report(1, 8, 256, 0)     # the batch=1 fault case for contrast
    report(64, 8, 1024, 0)   # a "passes/small-err" ctx for contrast
