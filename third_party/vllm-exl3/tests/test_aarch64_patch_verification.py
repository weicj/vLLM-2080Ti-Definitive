"""Regression coverage for the CI false positive on legitimate Linux guards."""

from pathlib import Path
import runpy

import pytest


VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/verify_exllamav3_aarch64.py"))
verify = VERIFY["verify"]
verify_pause_source = VERIFY["verify_pause_source"]

PAUSE = """#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
"""
SOURCE = """#include <thread>
#ifdef __linux__
#include <dlfcn.h>
#endif
inline void pause_cpu() {
""" + PAUSE + """}
#ifdef __linux__
void *load_cuda() { return dlopen("libcuda.so.1", RTLD_LAZY); }
#endif
"""


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("indent", ["", "                "])
def test_valid_linux_guards_survive_pause_verification(newline, indent):
    source = "\n".join(indent + line for line in SOURCE.splitlines())
    verify_pause_source(source.replace("\n", newline), "moe_handoff.cu")


@pytest.mark.parametrize(
    "broken",
    [SOURCE.replace("#if defined(__x86_64__) || defined(__i386__)", "#ifdef __linux__"),
     SOURCE.replace("std::this_thread::yield();", "_mm_pause();"),
     SOURCE + "\n__builtin_ia32_pause();\n",
     SOURCE + "\n_mm_pause();\n",
     SOURCE.replace("#include <thread>", ""),
     SOURCE.replace(PAUSE, "/*\n" + PAUSE + "*/\n")],
)
def test_unsafe_or_missing_pause_patch_is_rejected(broken):
    with pytest.raises(ValueError):
        verify_pause_source(broken, "moe_handoff.cu")


def test_verifier_requires_both_pause_files_and_disabled_feature_stubs(tmp_path):
    for name in VERIFY["PAUSE_FILES"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SOURCE, encoding="utf-8")
    (tmp_path / "avx2_target.cpp").write_text(
        "bool is_avx2_supported() { return false; }\n"
        "bool is_f16c_supported() { return false; }\n", encoding="utf-8",
    )
    avx512 = tmp_path / "avx512_target.cpp"
    avx512.write_text("bool is_avx512_supported() { return false; }\n", encoding="utf-8")
    verify(tmp_path)
    avx512.write_text("bool is_avx512_supported() { return true; }\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is_avx512_supported"):
        verify(tmp_path)
    avx512.write_text("bool is_avx512_supported() { return false; }\n", encoding="utf-8")
    (tmp_path / "parallel/all_reduce_cpu.cu").unlink()
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
