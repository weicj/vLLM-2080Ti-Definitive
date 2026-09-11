#!/usr/bin/env python3
"""Check the ARM64 source patch without rejecting unrelated Linux OS guards.

This verifies the expected pause blocks and CPU feature stubs, not a CUDA build.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


PAUSE_FILES = ("cpu/moe_handoff.cu", "parallel/all_reduce_cpu.cu")
PAUSE_BLOCK = re.compile(
    r"^[ \t]*#if[ \t]+defined\(__x86_64__\)[ \t]*\|\|[ \t]*defined\(__i386__\)[ \t]*\n"
    r"[ \t]*__builtin_ia32_pause\(\);[ \t]*\n"
    r"[ \t]*#else[ \t]*\n"
    r"[ \t]*std::this_thread::yield\(\);[ \t]*\n"
    r"[ \t]*#endif\b",
    re.MULTILINE,
)
PAUSE_CALL = re.compile(r"\b(?:__builtin_ia32_pause|_mm_pause)\s*\(")


def verify_pause_source(source: str, name: str) -> None:
    # Ignore comments while preserving string/character literals, so a commented
    # example of the patched block cannot satisfy the check.
    source = re.sub(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|//[^\n]*|/\*.*?\*/',
        lambda match: match.group(1) or "",
        source,
        flags=re.DOTALL,
    )
    source = source.replace("\r\n", "\n")
    remaining, count = PAUSE_BLOCK.subn("", source)
    if count != 1:
        raise ValueError(f"{name}: expected one x86-only pause block with a portable yield fallback")
    if PAUSE_CALL.search(remaining):
        raise ValueError(f"{name}: pause intrinsic outside the verified architecture guard")
    if not re.search(r"^\s*#include\s*<thread>", source, re.MULTILINE):
        raise ValueError(f"{name}: portable yield requires <thread>")


def verify(root: Path) -> None:
    for name in PAUSE_FILES:
        verify_pause_source((root / name).read_text(encoding="utf-8"), name)
    for name, functions in (
        ("avx2_target.cpp", ("is_avx2_supported", "is_f16c_supported")),
        ("avx512_target.cpp", ("is_avx512_supported",)),
    ):
        source = (root / name).read_text(encoding="utf-8")
        for function in functions:
            if not re.search(rf"bool\s+{function}\s*\(\s*\)\s*\{{\s*return\s+false\s*;\s*\}}", source):
                raise ValueError(f"{name}: missing disabled {function} stub")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exllamav3_ext_dir", type=Path)
    args = parser.parse_args()
    try:
        verify(args.exllamav3_ext_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    print("SUCCESS: x86 pause guards, portable yield, and disabled AVX feature stubs verified")


if __name__ == "__main__":
    main()
