#!/usr/bin/env python3
"""Compile each .sp3 with sp3 to CS and emit .text / .bin / .hex."""

from __future__ import annotations

import argparse
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_paren_field(text: str, field_name: str) -> str | None:
    """Parse the parenthesized value for `field_name(value)` from shader text (line-based, multiline allowed)."""
    pattern = rf"^\s*{re.escape(field_name)}\s*\(\s*([^)]*?)\s*\)"
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_kernel_text_fields(text_path: Path) -> dict[str, str | None]:
    if not text_path.is_file():
        print(f"Error: {text_path.name} not found; cannot parse.", file=sys.stderr)
        sys.exit(1)

    content = text_path.read_text(encoding="utf-8", errors="replace")

    fields = (
        "sgpr_count",
        "vgpr_count",
        "user_sgpr_count",
        "tidig_comp_cnt",
        "wave_size",
    )
    return {name: _parse_paren_field(content, name) for name in fields}


def print_kernel_text_fields(parsed: dict[str, str | None], text_path: Path) -> None:
    fields = (
        "sgpr_count",
        "vgpr_count",
        "user_sgpr_count",
        "tidig_comp_cnt",
        "wave_size",
    )
    print("--- Fields parsed from kernel.text (parenthesized values) ---", flush=True)
    for name in fields:
        value = parsed[name]
        if value is None:
            print(f"  {name}: <not found>", flush=True)
        else:
            print(f"  {name}: {value}", flush=True)
    print(f"(source: {text_path})", flush=True)


def ensure_kernel_s(kernel_s_path: Path) -> None:
    """If a same-named .s exists next to the .sp3, use it; else copy kernel_template.s from this script directory."""
    if kernel_s_path.is_file():
        print(f"--- Using existing {kernel_s_path.name} ({kernel_s_path.parent}) ---", flush=True)
        return
    template = Path(__file__).resolve().parent / "kernel_template.s"
    if not template.is_file():
        print(
            f"Error: {kernel_s_path.name} not found and template missing next to script: {template}",
            file=sys.stderr,
        )
        sys.exit(1)
    shutil.copy2(template, kernel_s_path)
    print(
        f"--- Copied {template.name} to {kernel_s_path} ---",
        flush=True,
    )


def patch_kernel_s(
    s_path: Path, sgpr_count: int, vgpr_count: int, lds_size: int
) -> None:
    """Update kernel.s fields from kernel.text GPR counts and LDS size."""
    if not s_path.is_file():
        print(f"Error: {s_path.name} not found; cannot update.", file=sys.stderr)
        sys.exit(1)

    text = s_path.read_text(encoding="utf-8", errors="replace")

    text = re.sub(
        r"(^[\t ]*\.amdhsa_next_free_vgpr\s+)\d+",
        lambda m: m.group(1) + str(vgpr_count),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.amdhsa_next_free_sgpr\s+)\d+",
        lambda m: m.group(1) + str(sgpr_count),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.sgpr_count:\s*)\d+",
        lambda m: m.group(1) + str(sgpr_count),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.vgpr_count:\s*)\d+",
        lambda m: m.group(1) + str(vgpr_count),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.amdhsa_group_segment_fixed_size\s+)\d+",
        lambda m: m.group(1) + str(lds_size),
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.group_segment_fixed_size:\s*)\d+",
        lambda m: m.group(1) + str(lds_size),
        text,
        flags=re.MULTILINE,
    )

    s_path.write_text(text, encoding="utf-8")
    print(f"--- Wrote {s_path} ---", flush=True)


def patch_cluster_config(
    s_path: Path, cluster_x: int, cluster_y: int
) -> None:
    """Set cluster .cluster_dims metadata in kernel.s for gfx12+.

    Cluster launch is configured at runtime via hipDrvLaunchKernelEx
    with hipLaunchAttributeClusterDimension; the .amdhsa_enable_*
    kernel descriptor directives are not supported on gfx1250.
    """
    if not s_path.is_file():
        print(f"Error: {s_path.name} not found; cannot patch cluster config.", file=sys.stderr)
        sys.exit(1)

    enable = cluster_x > 1 or cluster_y > 1
    text = s_path.read_text(encoding="utf-8", errors="replace")

    # --- .amdgpu_metadata YAML block ---
    cluster_dims_line = f"    .cluster_dims: [ {cluster_x}, {cluster_y}, 1 ]"
    # Replace existing .cluster_dims (commented or not)
    pat_cd = r"^[\t ]*;?[\t ]*\.cluster_dims:.*$"
    if re.search(pat_cd, text, re.MULTILINE):
        if enable:
            text = re.sub(pat_cd, cluster_dims_line, text, flags=re.MULTILINE)
        else:
            # Comment it out when no cluster
            text = re.sub(pat_cd, f"    ;.cluster_dims: [ 1, 1, 1 ]", text, flags=re.MULTILINE)
    elif enable:
        # Insert before .name: line
        text = re.sub(
            r"(^[\t ]+\.name:)",
            cluster_dims_line + "\n\\1",
            text,
            flags=re.MULTILINE,
        )

    s_path.write_text(text, encoding="utf-8")
    if enable:
        print(
            f"--- Patched cluster config: cluster_x={cluster_x}, cluster_y={cluster_y} in {s_path.name} ---",
            flush=True,
        )
    else:
        print(f"--- Cluster disabled (1x1) in {s_path.name} ---", flush=True)


def find_cpp_file(sp3_path: Path) -> Path | None:
    """Auto-discover the .cpp host code file near the .sp3."""
    for search_dir in (sp3_path.parent, sp3_path.parent.parent):
        cpps = list(search_dir.glob("*.cpp"))
        if len(cpps) == 1:
            return cpps[0]
    return None


def _parse_args_set_arg(content: str) -> list[dict]:
    """Parse kernel args from CSIM-style set_arg() calls."""
    access_map: dict[str, str] = {}
    for m in re.finditer(
        r'(\w+)\s*=\s*\w+->create_buffer\(\s*(TYPE_READ_WRITE|TYPE_READ)\b', content
    ):
        access_map[m.group(1)] = (
            'read_write' if m.group(2) == 'TYPE_READ_WRITE' else 'read_only'
        )

    pattern = re.compile(
        r'->set_arg\s*\(\s*(MEM|NONE_MEM)\s*,'
        r'\s*\w+\+\+\s*,'
        r'\s*sizeof\([^)]+\)\s*,'
        r'\s*\(void\*\)\s*'
        r'\(?&?(\w+)\)?'
        r'\s*\)'
    )

    args: list[dict] = []
    offset = 0
    for m in pattern.finditer(content):
        mem_type = m.group(1)
        var_name = m.group(2)

        if mem_type == 'MEM':
            args.append({
                'name': var_name,
                'size': 8,
                'offset': offset,
                'kind': 'global_buffer',
                'access': access_map.get(var_name, 'read_only'),
            })
        else:
            args.append({
                'name': var_name,
                'size': 4,
                'offset': offset,
                'kind': 'by_value',
                'access': None,
            })
        offset += 16

    return args


def _parse_args_packed_struct(content: str) -> list[dict]:
    """Parse kernel args from packed struct with p2/p3 padding.

    Pattern:
        void *name;      p2 _padN;   -> pointer (8B), global_buffer, 16B slot
        unsigned int name; p3 _padN;  -> scalar (4B), by_value, 16B slot
    Access is determined by hipMemcpy(..., hipMemcpyDeviceToHost) calls.
    """
    # Find the anonymous args struct block.
    # Handles plain `struct {` and `struct __attribute__((packed)) {`
    # but skips named structs like `struct p3 {`.
    struct_match = re.search(
        r'struct\s*(?:__attribute__\s*\(\([^)]*\)\)\s*)?\{(.*?)\}\s*args\s*;',
        content, re.DOTALL,
    )
    if not struct_match:
        return []

    struct_body = struct_match.group(1)

    # Find buffers read back from device (= read_write)
    readback_bufs: set[str] = set()
    for m in re.finditer(
        r'hipMemcpy\s*\([^,]+,\s*(\w+)\s*,.*?hipMemcpyDeviceToHost\)', content
    ):
        readback_bufs.add(m.group(1))

    # Map struct field name -> assigned buffer variable: args.ptr_X = (void*)XBuffer;
    field_to_buf: dict[str, str] = {}
    for m in re.finditer(r'args\.(\w+)\s*=\s*\(void\s*\*\)\s*(\w+)', content):
        field_to_buf[m.group(1)] = m.group(2)

    # Parse struct fields (skip padding fields)
    field_pat = re.compile(
        r'(?:void\s*\*\s*|unsigned\s+int\s+|uint32_t\s+|int\s+|float\s+|double\s+)(\w+)\s*;'
    )
    type_pat = re.compile(r'void\s*\*')

    args: list[dict] = []
    offset = 0
    for line in struct_body.splitlines():
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        fm = field_pat.search(line)
        if not fm:
            continue
        name = fm.group(1)
        if name.startswith('_p'):  # skip padding fields like _p0, _pad0
            continue

        is_ptr = bool(type_pat.search(line))
        if is_ptr:
            buf_var = field_to_buf.get(name, '')
            access = 'read_write' if buf_var in readback_bufs else 'read_only'
            args.append({
                'name': name,
                'size': 8,
                'offset': offset,
                'kind': 'global_buffer',
                'access': access,
            })
        else:
            args.append({
                'name': name,
                'size': 4,
                'offset': offset,
                'kind': 'by_value',
                'access': None,
            })
        offset += 16

    return args


def _parse_workgroup_size(content: str) -> int | None:
    """Parse workgroup size (bdx * bdy * bdz) from C++ host code."""
    dims = {}
    for var in ('bdx', 'bdy', 'bdz'):
        m = re.search(rf'\bint\s+{var}\s*=\s*(\d+)\s*;', content)
        if m:
            dims[var] = int(m.group(1))
    if 'bdx' not in dims:
        return None
    return dims.get('bdx', 1) * dims.get('bdy', 1) * dims.get('bdz', 1)


def parse_kernel_args_from_cpp(cpp_path: Path) -> list[dict]:
    """Parse kernel args from C++ host code.

    Supports two patterns:
    1. CSIM-style set_arg() calls (MEM/NONE_MEM)
    2. Silicon-style packed struct with p2/p3 padding
    """
    content = cpp_path.read_text(encoding="utf-8", errors="replace")

    args = _parse_args_set_arg(content)
    if not args:
        args = _parse_args_packed_struct(content)

    return args


def apply_kernarg_preload(s_path: Path, preload_len: int, kernarg_size: int) -> None:
    """Enable HW kernarg preload in the .amdhsa_kernel descriptor.

    - Injects .amdhsa_user_sgpr_kernarg_preload_length/offset so the CP preloads
      the first `preload_len` kernarg DWORDS into SGPRs after the kernarg segment
      ptr (s0:1) -> s2 .. s(1+preload_len).
    - Bumps .amdhsa_user_sgpr_count to 2 + preload_len (segment_ptr + preloaded).
    - Optionally overrides .amdhsa_kernarg_size / .kernarg_segment_size to the
      packed preload ABI size (e.g. 152 = 0x98), since the .cpp parser may have
      computed the legacy 16-byte-slot size.
    """
    text = s_path.read_text(encoding="utf-8", errors="replace")

    if kernarg_size > 0:
        text = re.sub(
            r'(^[\t ]*\.amdhsa_kernarg_size\s+)\d+',
            lambda m: m.group(1) + str(kernarg_size),
            text, flags=re.MULTILINE)
        text = re.sub(
            r'(^[\t ]*\.kernarg_segment_size:\s*)\d+',
            lambda m: m.group(1) + str(kernarg_size),
            text, flags=re.MULTILINE)

    # segment_ptr (2 SGPRs) + preloaded dwords
    text = re.sub(
        r'(^[\t ]*\.amdhsa_user_sgpr_count\s+)\d+',
        lambda m: m.group(1) + str(2 + preload_len),
        text, flags=re.MULTILINE)

    if 'kernarg_preload_length' not in text:
        text = re.sub(
            r'(^[\t ]*\.amdhsa_user_sgpr_kernarg_segment_ptr\s+1[ \t]*\n)',
            lambda m: (m.group(1)
                       + '\t.amdhsa_user_sgpr_kernarg_preload_length ' + str(preload_len) + '\n'
                       + '\t.amdhsa_user_sgpr_kernarg_preload_offset 0\n'),
            text, count=1, flags=re.MULTILINE)

    s_path.write_text(text, encoding="utf-8")
    ks = "(unchanged)" if kernarg_size == 0 else str(kernarg_size)
    print(f"--- PRELOAD: kernarg_preload_length={preload_len}, kernarg_size={ks}, "
          f"user_sgpr_count={2 + preload_len} ---", flush=True)


def patch_kernel_args(s_path: Path, cpp_path: Path) -> bool:
    """Update .args, .amdhsa_kernarg_size, .kernarg_segment_size,
    and .max_flat_workgroup_size in kernel.s based on C++ host code.
    Returns True if args were successfully patched."""
    content = cpp_path.read_text(encoding="utf-8", errors="replace")
    args = parse_kernel_args_from_cpp(cpp_path)
    if not args:
        print(
            f"Warning: no kernel args parsed from {cpp_path.name}; skipping.",
            file=sys.stderr,
        )
        return False

    # kernarg_segment_size = last arg offset + 16 (aligned to 16-byte slots)
    last_offset = max(a['offset'] for a in args)
    kernarg_size = last_offset + 16

    # workgroup size from bdx * bdy * bdz
    wg_size = _parse_workgroup_size(content)

    # Build .args YAML lines
    arg_lines: list[str] = []
    for a in args:
        if a['kind'] == 'global_buffer':
            access = a['access']
            pad = ' ' if access == 'read_only' else ''
            arg_lines.append(
                f"    - {{.value_kind: global_buffer, .offset: {a['offset']:>3d}, "
                f".size: {a['size']}, .actual_access: {access},{pad} .address_space: global}}"
                f"  ; {a['name']}"
            )
        else:
            arg_lines.append(
                f"    - {{.value_kind: by_value,      .offset: {a['offset']:>3d}, "
                f".size: {a['size']}}}"
                f"  ; {a['name']}"
            )

    text = s_path.read_text(encoding="utf-8", errors="replace")

    # 1) Replace .args entries (lines starting with '    - {' after '- .args:')
    args_block = "\n".join(arg_lines)
    text = re.sub(
        r'(^[ \t]*-[ \t]+\.args:\s*\n)(?:[ \t]*-[ \t]+\{.*\n)*',
        lambda m: m.group(1) + args_block + "\n",
        text,
        flags=re.MULTILINE,
    )

    # 2) Update .amdhsa_kernarg_size
    text = re.sub(
        r'(^[\t ]*\.amdhsa_kernarg_size\s+)\d+',
        lambda m: m.group(1) + str(kernarg_size),
        text,
        flags=re.MULTILINE,
    )

    # 3) Update .kernarg_segment_size
    text = re.sub(
        r'(^[\t ]*\.kernarg_segment_size:\s*)\d+',
        lambda m: m.group(1) + str(kernarg_size),
        text,
        flags=re.MULTILINE,
    )

    # 4) Update .max_flat_workgroup_size
    if wg_size is not None:
        text = re.sub(
            r'(^[\t ]*\.max_flat_workgroup_size:\s*)\d+',
            lambda m: m.group(1) + str(wg_size),
            text,
            flags=re.MULTILINE,
        )

    s_path.write_text(text, encoding="utf-8")
    wg_msg = f", workgroup_size={wg_size}" if wg_size else ""
    print(
        f"--- Patched kernel args ({len(args)} args, kernarg_size={kernarg_size}{wg_msg}) "
        f"in {s_path.name} from {cpp_path.name} ---",
        flush=True,
    )
    return True


def _hex_tokens_from_file(hex_path: Path) -> list[str]:
    if not hex_path.is_file():
        print(f"Error: {hex_path.name} not found.", file=sys.stderr)
        sys.exit(1)
    raw = hex_path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"0x[0-9a-fA-F]+", raw)


def embed_hex_as_long_directives(
    s_path: Path, hex_path: Path, kernel_name: str
) -> None:
    """Replace code between .amdhsa_code_object_version and first .section; emit .text header and .longs; sync kernel symbol."""
    tokens = _hex_tokens_from_file(hex_path)
    if not tokens:
        print(f"Error: no 0x… hex constants found in {hex_path.name}.", file=sys.stderr)
        sys.exit(1)

    while tokens and tokens[-1].lower() == "0x00000000":
        tokens.pop()
    if not tokens:
        print(f"Error: all hex constants in {hex_path.name} are 0x00000000.", file=sys.stderr)
        sys.exit(1)

    text = s_path.read_text(encoding="utf-8", errors="replace")

    text = re.sub(
        r"(^[\t ]*\.amdhsa_kernel)\s+\S+",
        rf"\1 {kernel_name}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.name:\s+)\S+",
        rf"\1{kernel_name}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^[\t ]*\.symbol:\s+)\S+",
        rf"\1{kernel_name}.kd",
        text,
        flags=re.MULTILINE,
    )

    lines = text.splitlines()

    anchor_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\.amdhsa_code_object_version\b", line):
            anchor_idx = i
            break

    if anchor_idx is None:
        print(
            "Error: .amdhsa_code_object_version line not found in kernel.s.",
            file=sys.stderr,
        )
        sys.exit(1)

    section_idx: int | None = None
    for j in range(anchor_idx + 1, len(lines)):
        if re.match(r"^\s*\.section\b", lines[j]):
            section_idx = j
            break

    if section_idx is not None:
        head = lines[: anchor_idx + 1]
        tail = lines[section_idx:]
    else:
        print(
            "Warning: no .section line; inserting after stripping standalone .long lines (template should include .section).",
            file=sys.stderr,
        )
        head = lines[: anchor_idx + 1]
        tail = [
            ln
            for ln in lines[anchor_idx + 1 :]
            if not re.match(r"^\s*\.long\b", ln)
        ]

    preamble = [
        "",
        ".text",
        f".globl    {kernel_name}",
        f".type     {kernel_name},@function",
        ".p2align  8",
        f"{kernel_name}:",
    ]
    long_lines = [f"\t.long {tok}" for tok in tokens]
    code_size_in_byte = len(tokens) * 4
    inst_pref_size = min(math.ceil(code_size_in_byte / 128), 255)
    padding_lines = [
        f"; codeLenInByte = {code_size_in_byte}",
        "\t.p2alignl 7, 0xbf9f0000",
        "\t.fill 96, 4, 0xbf9f0000",
    ]

    new_lines = head + preamble + long_lines + padding_lines + [""] + tail
    final_text = "\n".join(new_lines) + "\n"

    final_text = re.sub(
        r"(^[\t ]*\.amdhsa_inst_pref_size\s+)\d+",
        lambda m: m.group(1) + str(inst_pref_size),
        final_text,
        flags=re.MULTILINE,
    )

    s_path.write_text(final_text, encoding="utf-8")
    cut_note = (
        " (removed old code between version and .section including all .long)"
        if section_idx is not None
        else " (removed all .long lines)"
    )
    print(
        f"--- Wrote kernel {kernel_name!r}: {len(long_lines)} .long lines from {hex_path.name}"
        f" (code_size={code_size_in_byte} bytes, inst_pref_size={inst_pref_size}){cut_note} ---",
        flush=True,
    )


def _extract_author_lines_from_sp3(sp3_path: Path) -> list[str]:
    """Extract lines containing the substring "Author" from .sp3 (order preserved)."""
    raw = sp3_path.read_text(encoding="utf-8", errors="replace")
    return [ln for ln in raw.splitlines() if "Author" in ln]


def _sp3_author_line_to_asm_comment(line: str) -> str:
    """Map .sp3 comment prefix to assembly ';' (supports // and line-leading * block style)."""
    s = line.rstrip("\n\r")
    m = re.match(r"^(\s*)//\s?(.*)$", s)
    if m:
        return f"{m.group(1)};{m.group(2)}"
    m2 = re.match(r"^(\s*)\*\s?(.*)$", s)
    if m2:
        return f"{m2.group(1)};{m2.group(2)}"
    m3 = re.match(r"^(\s*)(.*)$", s)
    return f"{m3.group(1)};{m3.group(2)}"


def sync_author_comments_from_sp3(sp3_path: Path, s_path: Path) -> None:
    """Collect Author lines from .sp3, drop old Author lines in kernel.s, convert to ';', prepend to file."""
    if not s_path.is_file():
        print(f"Error: {s_path.name} not found; cannot sync Author comments.", file=sys.stderr)
        sys.exit(1)

    author_sp3_lines = _extract_author_lines_from_sp3(sp3_path)
    body = s_path.read_text(encoding="utf-8", errors="replace")
    kept_lines = [ln for ln in body.splitlines() if "Author" not in ln]

    if author_sp3_lines:
        asm_author = [_sp3_author_line_to_asm_comment(ln) for ln in author_sp3_lines]
        new_text = "\n".join(asm_author + kept_lines) + "\n"
        s_path.write_text(new_text, encoding="utf-8")
        print(
            f"--- Synced {len(asm_author)} Author comment lines from {sp3_path.name} to top of {s_path.name} ---",
            flush=True,
        )
    else:
        new_text = "\n".join(kept_lines) + "\n"
        s_path.write_text(new_text, encoding="utf-8")
        print(
            f"--- No Author lines in {sp3_path.name}; removed any existing Author lines from {s_path.name} ---",
            flush=True,
        )


def print_kernel_s_kernarg_line_hints(s_path: Path) -> None:
    """After kernel.s is generated, print 1-based line hints for kernarg sizes and .args."""
    if not s_path.is_file():
        return
    lines = s_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def first_line_no(pattern: str) -> int | None:
        rx = re.compile(pattern)
        for i, ln in enumerate(lines, start=1):
            if rx.search(ln):
                return i
        return None

    ln_amd = first_line_no(r"\.amdhsa_kernarg_size\b")
    ln_kseg = first_line_no(r"\.kernarg_segment_size\b")
    ln_args = first_line_no(r"-\s+\.args\s*:")

    def hint(label: str, n: int | None) -> str:
        if n is not None:
            return f"{label} at line {n}"
        return f"{label} not found"

    print(
        "PASSED\n"
        + "Please manually edit kernel input argument size: "
        + hint(".amdhsa_kernarg_size", ln_amd)
        + "; "
        + hint(".kernarg_segment_size", ln_kseg),
        flush=True,
    )
    print(
        "Please manually edit kernel input argument list: "
        + hint(".args:", ln_args),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sp3 -allow-raw-bits compile (type=CS) on the given .sp3 and write text/binary/hex."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FILE.sp3",
        help="Path to input .sp3 file.",
    )
    parser.add_argument(
        "--sp3",
        default="sp3",
        metavar="EXE",
        help="Path to sp3 executable (default: find sp3 in PATH).",
    )
    parser.add_argument(
        "-n",
        "--kernel-name",
        default=None,
        metavar="NAME",
        help="Kernel symbol name; default: basename of input .sp3 (without .sp3).",
    )
    parser.add_argument(
        "-lds",
        "--lds-size",
        type=int,
        required=True,
        metavar="BYTES",
        help="LDS size in bytes for .amdhsa_group_segment_fixed_size and "
        ".group_segment_fixed_size.",
    )
    parser.add_argument(
        "-s",
        "--sgpr",
        type=int,
        required=True,
        metavar="N",
        help="SGPR count.",
    )
    parser.add_argument(
        "-v",
        "--vgpr",
        type=int,
        required=True,
        metavar="N",
        help="VGPR count.",
    )
    parser.add_argument(
        "--cluster-x",
        type=int,
        default=1,
        metavar="N",
        help="Cluster X dimension in workgroups (default: 1, no cluster).",
    )
    parser.add_argument(
        "--cluster-y",
        type=int,
        default=1,
        metavar="N",
        help="Cluster Y dimension in workgroups (default: 1, no cluster).",
    )
    parser.add_argument(
        "-c",
        "--cpp",
        default=None,
        metavar="FILE.cpp",
        help="Path to C++ host code for parsing kernel args (set_arg calls). "
        "Default: auto-discover *.cpp near the .sp3 file.",
    )
    parser.add_argument(
        "--preload",
        type=int,
        default=0,
        metavar="N",
        help="HW kernarg preload length in DWORDS (0 = off / legacy ABI). When >0, "
        "emits .amdhsa_user_sgpr_kernarg_preload_length/offset and bumps "
        ".amdhsa_user_sgpr_count to 2+N (segment_ptr + preloaded dwords).",
    )
    parser.add_argument(
        "--kernarg-size",
        type=int,
        default=0,
        metavar="BYTES",
        help="Override .amdhsa_kernarg_size and .kernarg_segment_size (e.g. 152 for "
        "the 0x98 packed preload ABI). 0 = keep the value parsed from the .cpp.",
    )
    args = parser.parse_args()

    sp3_src = Path(args.input).expanduser().resolve()
    if not sp3_src.is_file():
        print(f"Error: input file not found: {sp3_src}", file=sys.stderr)
        sys.exit(1)

    kernel = sp3_src.stem
    kernel_name = args.kernel_name if args.kernel_name is not None else kernel
    out_dir = sp3_src.parent

    write_specs = (
        ("-text", out_dir / f"{kernel}.text"),
        ("-binary", out_dir / f"{kernel}.bin"),
        ("-hex", out_dir / f"{kernel}.hex"),
    )

    total = len(write_specs)
    for step, (out_flag, out_path) in enumerate(write_specs, start=1):
        cmd = [
            args.sp3,
            "compile:",
            "-allow-raw-bits",
            "asic=MI450",
            "type=CS",
            str(sp3_src),
            "write:",
            out_flag,
            str(out_path),
        ]
        # shlex.join exists in Python 3.8+; keep compatible with older runtimes.
        cmd_line = (
            shlex.join(cmd)
            if hasattr(shlex, "join")
            else " ".join(shlex.quote(str(x)) for x in cmd)
        )
        print(f"[Step {step}/{total}] Running:\n  {cmd_line}\n", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: command failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)

    kernel_text_path = out_dir / f"{kernel}.text"
    parsed = parse_kernel_text_fields(kernel_text_path)
    print_kernel_text_fields(parsed, kernel_text_path)

    sgpr_n = args.sgpr
    vgpr_n = args.vgpr

    kernel_s_path = out_dir / f"{kernel}.s"
    ensure_kernel_s(kernel_s_path)
    print(
        f"--- Updating kernel.s from parsed results (LDS={args.lds_size} bytes) ---",
        flush=True,
    )
    patch_kernel_s(kernel_s_path, sgpr_n, vgpr_n, args.lds_size)
    patch_cluster_config(kernel_s_path, args.cluster_x, args.cluster_y)

    kernel_hex_path = out_dir / f"{kernel}.hex"
    print(
        f"--- Expanding kernel.hex into .long and setting kernel name {kernel_name!r} ---",
        flush=True,
    )
    embed_hex_as_long_directives(kernel_s_path, kernel_hex_path, kernel_name)

    args_patched = False
    if args.cpp:
        cpp_path = Path(args.cpp).expanduser().resolve()
    else:
        cpp_path = find_cpp_file(sp3_src)
    if cpp_path and cpp_path.is_file():
        print(f"--- Patching kernel args from {cpp_path.name} ---", flush=True)
        args_patched = patch_kernel_args(kernel_s_path, cpp_path)
    else:
        print("Warning: no .cpp file found; skipping kernel args patch.", file=sys.stderr)

    sync_author_comments_from_sp3(sp3_src, kernel_s_path)

    if args.preload > 0:
        apply_kernarg_preload(kernel_s_path, args.preload, args.kernarg_size)

    if args_patched:
        print("PASSED", flush=True)
    else:
        print_kernel_s_kernarg_line_hints(kernel_s_path)


if __name__ == "__main__":
    main()
