#!/usr/bin/env python3
"""Run reproducible Qwen Flash-Next throughput samples and write an audit manifest.

The server is intentionally managed outside this helper.  This keeps the
benchmark safe for an existing deployment while recording enough metadata to
tell a real result from a load-only or incomplete request.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((4096, 128, "4k128"), (32768, 512, "32k512"))
PROFILE_KEYS = (
    "SERVED_NAME",
    "MODE",
    "MODEL_FAMILY",
    "PROFILE_GROUP",
    "MODEL_VARIANT",
    "TP_SIZE",
    "PP_SIZE",
    "VLLM_PP_LAYER_PARTITION",
    "QUANTIZATION",
    "KV_CACHE_DTYPE",
    "MAX_MODEL_LEN",
    "GPU_UTIL",
    "MAX_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "NO_ASYNC_SCHEDULING",
    "DISABLE_PREFIX_CACHING",
    "MTP_K",
    "SPECULATIVE_METHOD",
    "SPECULATIVE_TOKENS",
    "MESSAGE_TYPE",
    "LANGUAGE_MODEL_ONLY",
    "SKIP_MM_PROFILING",
    "CUSTOM_ALL_REDUCE_MODE",
    "ADDITIONAL_CONFIG_JSON",
    "VLLM_FORCE_NVFP4_W4A16",
    "PLE_PLACEMENT",
)


def profile_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line or not line.split("=", 1)[0].replace("_", "").isalnum():
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_metadata() -> dict[str, Any]:
    return {
        "commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "branch": command_output(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "dirty": bool(
            command_output(["git", "-C", str(ROOT), "status", "--porcelain"])
        ),
    }


def gpu_metadata() -> dict[str, Any]:
    raw = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            rows.append(
                dict(
                    zip(
                        (
                            "index",
                            "name",
                            "compute_cap",
                            "memory_total_mib",
                            "driver_version",
                        ),
                        fields,
                    )
                )
            )
    return {"query": "index,name,compute_cap,memory.total,driver_version", "gpus": rows}


def extract_record(path: Path) -> dict[str, Any]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"benchmark helper produced no record: {path}")
    record = json.loads(lines[-1])
    if record.get("http_status") != 200 or not record.get("stream_done"):
        raise RuntimeError(f"request did not complete: {record}")
    if record.get("prompt_tokens") != record.get("requested_prompt_tokens"):
        raise RuntimeError(f"prompt token contract failed: {record}")
    if record.get("completion_tokens") != record.get("requested_completion_tokens"):
        raise RuntimeError(f"completion token contract failed: {record}")
    for key in ("prefill_tok_s", "decode_tok_s"):
        if not isinstance(record.get(key), (int, float)) or record[key] <= 0:
            raise RuntimeError(f"missing positive {key}: {record}")
    return record


def run_sample(
    args: argparse.Namespace,
    prompt_tokens: int,
    gen_tokens: int,
    label: str,
    out: Path,
    gpu_log: Path,
) -> dict[str, Any]:
    command = [
        args.python,
        str(ROOT / "tools/profile_request.py"),
        "--model-dir",
        args.model_dir,
        "--served-name",
        args.served_name,
        "--base-url",
        args.base_url,
        "--endpoint",
        "completions",
        "--prompt-tokens",
        str(prompt_tokens),
        "--gen-tokens",
        str(gen_tokens),
        "--label",
        label,
        "--prompt-variant",
        label,
        "--out",
        str(out),
        "--gpu-log",
        str(gpu_log),
        "--ignore-eos",
        "--pure-filler",
        "--allowed-token-text",
        " the",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} timed out after {args.timeout}s") from exc
    if result.returncode:
        raise RuntimeError(
            f"{label} exited {result.returncode}: {result.stderr[-2000:]}"
        )
    return extract_record(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--profile", type=Path, required=True, help="route profile .env"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="audit manifest JSON"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter containing the runtime benchmark dependencies",
    )
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    profile = args.profile if args.profile.is_absolute() else ROOT / args.profile
    if not profile.is_file():
        raise SystemExit(f"profile not found: {profile}")
    values = profile_values(profile)
    for key, expected in (
        ("MTP_K", "0"),
        ("SPECULATIVE_METHOD", "none"),
        ("PLE_PLACEMENT", "disk"),
    ):
        if values.get(key) != expected:
            raise SystemExit(
                f"profile must set {key}={expected}, got {values.get(key)!r}"
            )
    kv_dtype = values.get("KV_CACHE_DTYPE", "").lower()
    if kv_dtype not in {"float16", "fp8"}:
        raise SystemExit(
            "profile must set KV_CACHE_DTYPE=float16 or fp8, "
            f"got {values.get('KV_CACHE_DTYPE')!r}"
        )
    if values.get("SERVED_NAME") != args.served_name:
        profile_served_name = values.get("SERVED_NAME")
        raise SystemExit(
            f"--served-name must match profile SERVED_NAME={profile_served_name!r}"
        )
    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema": "qwenflash-audit/v1",
        "started_at": started,
        "profile": str(profile.relative_to(ROOT)),
        "profile_values": {
            key: values.get(key) for key in PROFILE_KEYS if key in values
        },
        "model_dir": os.path.abspath(args.model_dir),
        "served_name": args.served_name,
        "base_url": args.base_url,
        "git": git_metadata(),
        "gpu": gpu_metadata(),
        "benchmark_python": os.path.abspath(args.python),
        "kv_cache_dtype": kv_dtype,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "VLLM_PLE_PLACEMENT",
                "VLLM_PLE_CPU_OFFLOAD",
                "VLLM_PP_LAYER_PARTITION",
                "TP_SIZE",
                "PP_SIZE",
                "MTP_K",
                "VLLM_FORCE_NVFP4_W4A16",
                "CUDA_HOME",
                "TORCH_CUDA_ARCH_LIST",
            )
            if os.environ.get(key) is not None
        },
        "contract": {
            "warmup": 1,
            "samples": 3,
            "shapes": [
                {"prompt_tokens": p, "completion_tokens": c, "name": n}
                for p, c, n in SHAPES
            ],
        },
        "results": {},
    }
    with tempfile.TemporaryDirectory(prefix="qwenflash-audit-") as temp:
        temp_path = Path(temp)
        for prompt_tokens, gen_tokens, shape in SHAPES:
            shape_records = []
            for sample in range(4):
                kind = "warmup" if sample == 0 else f"sample{sample}"
                record = run_sample(
                    args,
                    prompt_tokens,
                    gen_tokens,
                    f"{shape}-{kind}",
                    temp_path / f"{shape}.jsonl",
                    temp_path / f"{shape}.gpu.log",
                )
                shape_records.append({"kind": kind, "record": record})
            samples = [entry["record"] for entry in shape_records[1:]]
            manifest["results"][shape] = {
                "warmup": shape_records[0],
                "samples": shape_records[1:],
                "summary": {
                    "prefill_tok_s_mean": statistics.mean(
                        r["prefill_tok_s"] for r in samples
                    ),
                    "prefill_tok_s_median": statistics.median(
                        r["prefill_tok_s"] for r in samples
                    ),
                    "decode_tok_s_mean": statistics.mean(
                        r["decode_tok_s"] for r in samples
                    ),
                    "decode_tok_s_median": statistics.median(
                        r["decode_tok_s"] for r in samples
                    ),
                },
            }
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "git": manifest["git"],
                "profile": manifest["profile"],
                "shapes": list(manifest["results"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
