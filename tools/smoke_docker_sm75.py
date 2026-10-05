#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SM75 image checks: metadata, GPU-less CLI help, and opt-in GPU kernels."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

# The repository supports Python 3.10, so avoid `tomllib` (3.11+). Only one
# scalar is needed out of pyproject.toml.
_FALLBACK_VERSION = re.compile(
    r'^\s*fallback_version\s*=\s*"([^"]+)"\s*$', re.MULTILINE
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def release_policy(bundle: Path) -> tuple[dict[str, str], str]:
    release = {}
    for line in (bundle / "PROJECT_RELEASE.env").read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts:
            key, value = parts[0].split("=", 1)
            release[key] = value
    match = _FALLBACK_VERSION.search(
        (bundle / "pyproject.toml").read_text(encoding="utf-8")
    )
    require(match is not None, "pyproject.toml has no setuptools_scm fallback_version")
    return release, match.group(1)


def check_versions(release: dict[str, str], expected: str, torch, vllm) -> None:
    require(
        torch.__version__ == release["VALIDATED_TORCH_VERSION"],
        f"Torch {torch.__version__} != {release['VALIDATED_TORCH_VERSION']}",
    )
    require(
        torch.version.cuda == release["VALIDATED_CUDA_VERSION"],
        f"Torch CUDA {torch.version.cuda} != {release['VALIDATED_CUDA_VERSION']}",
    )
    require(vllm.__version__ == expected, f"vLLM {vllm.__version__} != {expected}")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    require(
        python == release["PRIMARY_PYTHON_VERSION"],
        f"Python {python} != {release['PRIMARY_PYTHON_VERSION']}",
    )


def check_flashqla(root: Path) -> Path:
    package = root / "flash_qla"
    legacy = package / "ops/gated_delta_rule/legacy/sm_legacy.py"
    required = {
        legacy: (
            "def _load_ext(",
            "def _validate_extension(",
            '"gdn_forward_varlen"',
            "def chunk_gated_delta_rule_fwd_legacy(",
            "def chunk_gated_delta_rule_fwd_legacy_varlen(",
        ),
        legacy.parent / "csrc/gdn_forward.cu": (
            'm.def("gdn_forward"',
            'm.def("gdn_forward_varlen"',
            "cu_seqlens",
        ),
    }
    # Inspect the import guards without importing the SM90/TileLang package.
    for name in ("__init__.py", "ops/__init__.py", "ops/gated_delta_rule/__init__.py"):
        required[package / name] = (
            "except (ImportError, OSError, RuntimeError, ValueError):",
        )
    for path, symbols in required.items():
        text = path.read_text(encoding="utf-8")
        for symbol in symbols:
            require(symbol in text, f"Missing FlashQLA patch {symbol!r} in {path}")
    return legacy


def cpu_smoke():
    bundle = Path(__file__).resolve().parent
    release, expected = release_policy(bundle)
    # CUDA-enabled torch itself imports without libcuda. Do not query torch.cuda,
    # import FlashInfer/FlashQLA, or load native vLLM kernels on the build runner.
    torch = importlib.import_module("torch")
    vllm = importlib.import_module("vllm")
    check_versions(release, expected, torch, vllm)
    require(importlib.metadata.version("vllm") == expected, "vLLM wheel version drift")
    files = importlib.metadata.distribution("vllm").files or []
    # CUDA extension names from setup.py and platforms/cuda.py, not the CPU _C.
    for name in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
        require(
            any(
                str(p).startswith(f"vllm/{name}.") and str(p).endswith(".so")
                for p in files
            ),
            f"vLLM wheel has no {name} extension artifact",
        )
    print(
        f"Metadata OK: vLLM={expected}, Torch={torch.__version__}, "
        f"CUDA={torch.version.cuda}, "
        f"FlashInfer={importlib.metadata.version('flashinfer-python')}",
        flush=True,
    )
    root = Path(os.environ["FLASHQLA_ROOT"])
    require(root.is_absolute(), "FLASHQLA_ROOT must be absolute")
    require(
        Path(os.environ["FLASHQLA_DIR"]).resolve() == root.resolve(),
        "FLASHQLA_DIR and FLASHQLA_ROOT disagree",
    )
    source = check_flashqla(root)
    require(
        Path(os.environ["TORCH_EXTENSIONS_DIR"]).is_absolute(),
        "TORCH_EXTENSIONS_DIR must be an absolute persistent-cache path",
    )
    for compiler in ("nvcc", "c++", "ninja"):
        require(shutil.which(compiler) is not None, f"Missing JIT tool: {compiler}")
    subprocess.run(
        [sys.executable, str(bundle / "check_torch_inductor_e8m0.py")], check=True
    )
    print(
        "CPU smoke passed: metadata, Torch patch, FlashQLA source, JIT tools. "
        "GPU execution and JIT compilation NOT validated.",
        flush=True,
    )
    return torch, source


def check_close(torch, actual, expected, *, atol=3e-3, rtol=3e-3) -> None:
    require(bool(torch.isfinite(actual).all()), "Kernel returned non-finite values")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def gpu_smoke(torch, source: Path) -> None:
    require(torch.cuda.is_available(), "--gpu requires an NVIDIA GPU and driver")
    require(torch.cuda.get_device_capability(0) == (7, 5), "--gpu requires SM75")
    torch.cuda.set_device(0)
    print(
        f"GPU smoke: cuda:0, {torch.cuda.get_device_name(0)}; "
        "single-device test, no TP/NVLink or performance claim",
        flush=True,
    )
    torch.manual_seed(0)
    importlib.import_module("vllm._C_stable_libtorch")
    x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    out = torch.empty(4, 64, device="cuda", dtype=torch.float16)
    torch.ops._C.silu_and_mul(out, x)
    check_close(torch, out, torch.nn.functional.silu(x[:, :64]) * x[:, 64:])
    print("vLLM CUDA core silu_and_mul passed", flush=True)

    # FlashInfer's CUDA-core decode supports Turing; its tensor-core/FA2 branch
    # is not a requirement for this smoke. Compare against FP32 attention.
    flashinfer = importlib.import_module("flashinfer")
    q = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(16, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    out = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout="NHD", use_tensor_cores=False
    )
    k_ref, v_ref = (t.float().repeat_interleave(2, dim=1) for t in (k, v))
    scores = torch.einsum("hd,nhd->hn", q.float(), k_ref) * 64**-0.5
    reference = torch.einsum("hn,nhd->hd", scores.softmax(-1), v_ref)
    check_close(torch, out, reference.to(out.dtype))
    print("FlashInfer CUDA-core decode attention passed (not FA2)", flush=True)

    # Load the same legacy source module as qwen_gdn_linear_attn.py and the
    # FlashQLA tools tests; do not import the SM90 package entrypoint.
    spec = importlib.util.spec_from_file_location("sm75_smoke_legacy", source)
    require(spec is not None and spec.loader is not None, f"Cannot load {source}")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy._load_ext()  # JIT and required gdn_forward/gdn_forward_varlen symbols.
    q = torch.nn.functional.normalize(
        torch.randn(1, 7, 2, 32, device="cuda"), dim=-1
    ).half()
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).half()
    v = (0.1 * torch.randn(1, 7, 4, 32, device="cuda")).half()
    g = -torch.nn.functional.softplus(torch.randn(1, 7, 4, device="cuda"))
    beta = torch.sigmoid(torch.randn_like(g))
    state = 0.01 * torch.randn(2, 4, 32, 32, device="cuda")
    outputs, states = [], []
    # Existing native-QKV acceptance: FP16 q/k/v, FP32 gates/state must match
    # FP32 staging exactly after the output cast (benchmark_flashqla_qkv_storage).
    for i, (start, end) in enumerate(((0, 3), (3, 7))):
        tensors = [t[:, start:end].contiguous() for t in (q, k, v, g, beta)]
        initial = state[i : i + 1].clone()
        out, final = legacy.chunk_gated_delta_rule_fwd_legacy(
            *tensors, initial_state=initial.clone()
        )
        control, control_state = legacy.chunk_gated_delta_rule_fwd_legacy(
            *(t.float() for t in tensors), initial_state=initial.clone()
        )
        check_close(torch, out, control.to(out.dtype), atol=0, rtol=0)
        check_close(torch, final, control_state, atol=0, rtol=0)
        outputs.append(out)
        states.append(final)
    offsets = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    out, final = legacy.chunk_gated_delta_rule_fwd_legacy_varlen(
        q, k, v, g, beta, offsets, initial_state=state.clone()
    )
    check_close(torch, out, torch.cat(outputs, dim=1), atol=0, rtol=0)
    check_close(torch, final, torch.cat(states, dim=0), atol=0, rtol=0)
    torch.cuda.synchronize()
    print(
        "FlashQLA JIT dense/native-QKV and packed-varlen outputs/state passed",
        flush=True,
    )


def cli_help_smoke() -> None:
    """Exercise the installed CLI, emulating only GPU-less help defaults."""
    console_script = shutil.which("vllm")
    require(console_script is not None, "Installed vllm console script missing")
    # CLI imports inspect argv, so set it before importing the platform module.
    with patch.object(sys, "argv", [console_script, "serve", "--help"]):
        platforms = importlib.import_module("vllm.platforms")
        platform = platforms.current_platform
        require(platform.is_unspecified(), "CLI smoke expects a GPU-less CUDA image")
        print(
            "Installed CLI help check with emulated device metadata only; "
            "this is NOT GPU startup validation.",
            flush=True,
        )
        # DeviceConfig's default factory requires a device while constructing
        # help. Keep UnspecifiedPlatform; do not select CPU kernels or change
        # the image environment. The same override is used in config tests.
        with patch.object(platform, "device_type", "cpu"):
            runpy.run_path(console_script, run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gpu",
        action="store_true",
        help="also execute CUDA/FlashInfer/FlashQLA on SM75 device 0",
    )
    mode.add_argument(
        "--cli-help",
        action="store_true",
        help="only run installed CLI help with emulated GPU-less device metadata",
    )
    args = parser.parse_args(argv)
    if args.cli_help:
        cli_help_smoke()
        return 0
    torch, source = cpu_smoke()
    if args.gpu:
        with torch.inference_mode():
            gpu_smoke(torch, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
