#!/usr/bin/env python3
"""Disable x86-only ExLlamaV3 CPU paths for its ARM64 CUDA build.

Replaces x86-only inline assembly (__builtin_ia32_pause and _mm_pause) with
std::this_thread::yield(), and stubs out AVX2 / AVX-512 target functions so
exllamav3_ext compiles cleanly on NVIDIA Grace / DGX Spark (aarch64).
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


def write(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    print("patched", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "exllamav3_ext_dir",
        type=Path,
        help="Path to exllamav3/exllamav3_ext directory containing bindings.cpp",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply patches even if current host machine is not aarch64 (for testing/cross-compilation)",
    )
    args = parser.parse_args()

    machine = platform.machine().lower()
    if not args.force and machine not in {"aarch64", "arm64"}:
        raise SystemExit(
            f"refusing ARM64 patch on host architecture '{machine}'. Pass --force to bypass."
        )

    root = args.exllamav3_ext_dir.resolve()
    if not (root / "bindings.cpp").is_file():
        raise SystemExit(f"not an ExLlamaV3 extension source directory: {root}")

    write(
        root / "avx2_target.h",
        """#pragma once
bool is_avx2_supported();
bool is_f16c_supported();
#define AVX2_TARGET
#define AVX2_F16C_TARGET
#define AVX2_TARGET_OPTIONAL
""",
    )
    write(
        root / "avx512_target.h",
        """#pragma once
bool is_avx512_supported();
#define AVX512_TARGET
#define AVX512_TARGET_OPTIONAL
""",
    )
    write(
        root / "avx2_target.cpp",
        """#include "avx2_target.h"
bool is_avx2_supported() { return false; }
bool is_f16c_supported() { return false; }
""",
    )
    write(
        root / "avx512_target.cpp",
        """#include "avx512_target.h"
bool is_avx512_supported() { return false; }
""",
    )
    write(
        root / "parallel/all_reduce_cpu_avx2.cpp",
        """#include "all_reduce_cpu_avx2.h"
#include <cstdlib>
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void cpu_reduce_parallel(
    void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*)(uint16_t*, const uint16_t*, size_t),
    uint16_t*, const uint16_t*, const uint16_t*, size_t, int
) { std::abort(); }
""",
    )
    write(
        root / "parallel/all_reduce_cpu_avx512.cpp",
        """#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) { std::abort(); }
void bf16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }
void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) { std::abort(); }
void fp16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
""",
    )
    write(
        root / "cpu/moe_mul1.cpp",
        """#include "moe_mul1.h"
#include <cstdlib>
int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    int64_t, double, int64_t
) { std::abort(); }
void exl3_moe_cpu_free_layer(int64_t) {}
void exl3_moe_cpu_forward(
    int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t
) { std::abort(); }
void exl3_moe_cpu_forward_raw(
    int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int
) { std::abort(); }
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) { std::abort(); }
void exl3_moe_cpu_set_prof(bool) {}
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
""",
    )

    for relative in ("cpu/moe_handoff.cu", "parallel/all_reduce_cpu.cu"):
        path = root / relative
        if not path.is_file():
            print(f"skipping {relative} (file not present)")
            continue
        source = path.read_text(encoding="utf-8")
        old = """#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif"""
        new = """#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif"""
        if relative.endswith("all_reduce_cpu.cu"):
            old = """#ifdef __linux__
                    __builtin_ia32_pause();
                #else
                    _mm_pause();
                #endif"""
            new = """#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                #else
                    std::this_thread::yield();
                #endif"""
        if source.count(old) == 1:
            write(path, source.replace(old, new))
        elif new in source:
            print("already patched", path)
        else:
            raise SystemExit(f"expected one x86 pause block in {path}")


if __name__ == "__main__":
    main()
