#!/usr/bin/env python3
"""Compare FP32-staged and native-QKV SM75 FlashQLA GDN forward paths.

The candidate leaves gate, beta, and recurrent state in FP32.  It is accepted
only when output and final state match the FP32-staged control exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def load_legacy(root: Path):
    source = root / "flash_qla" / "ops" / "gated_delta_rule" / "legacy" / "sm_legacy.py"
    spec = importlib.util.spec_from_file_location("flashqla_bench_legacy", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed_ms(fn: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[round((len(ordered) - 1) * 0.9)],
        "min_ms": ordered[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flashqla-root",
        type=Path,
        default=Path(os.environ["FLASHQLA_ROOT"]),
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.tokens % args.sequences:
        raise ValueError("tokens must be divisible by sequences")

    torch.manual_seed(20260912)
    torch.cuda.manual_seed_all(20260912)
    device = torch.device("cuda")
    tokens_per_sequence = args.tokens // args.sequences
    q = F.normalize(
        torch.randn((1, args.tokens, 16, 128), device=device), dim=-1
    ).to(torch.float16)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(
        torch.float16
    )
    v = (0.1 * torch.randn((1, args.tokens, 48, 128), device=device)).to(
        torch.float16
    )
    # GDN uses a log-decay gate. A positive random gate creates an unstable
    # synthetic recurrence rather than a representative model input.
    g = -F.softplus(torch.randn((1, args.tokens, 48), device=device))
    beta = torch.sigmoid(torch.randn_like(g))
    state = 0.01 * torch.randn(
        (args.sequences, 48, 128, 128), device=device, dtype=torch.float32
    )
    scale = 128 ** -0.5
    legacy = load_legacy(args.flashqla_root)

    g32, beta32, state32 = (x.contiguous() for x in (g, beta, state))

    def fp32_staged(
        forward: Callable[..., tuple[torch.Tensor, torch.Tensor]], *forward_args
    ):
        output, final_state = forward(
            q.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            *forward_args,
        )
        return output.to(torch.float16), final_state

    def native_qkv(
        forward: Callable[..., tuple[torch.Tensor, torch.Tensor]], *forward_args
    ):
        output, final_state = forward(q, k, v, *forward_args)
        return output.to(torch.float16), final_state

    control_state = state32.clone()
    candidate_state = state32.clone()
    if args.sequences == 1:
        control = lambda: fp32_staged(
            legacy.chunk_gated_delta_rule_fwd_legacy,
            g32,
            beta32,
            scale,
            control_state,
        )
        candidate = lambda: native_qkv(
            legacy.chunk_gated_delta_rule_fwd_legacy,
            g32,
            beta32,
            scale,
            candidate_state,
        )
        layout = "dense"
    else:
        offsets = torch.arange(
            0, args.tokens + 1, tokens_per_sequence, device=device, dtype=torch.int32
        )
        control = lambda: fp32_staged(
            legacy.chunk_gated_delta_rule_fwd_legacy_varlen,
            g32,
            beta32,
            offsets,
            scale,
            control_state,
        )
        candidate = lambda: native_qkv(
            legacy.chunk_gated_delta_rule_fwd_legacy_varlen,
            g32,
            beta32,
            offsets,
            scale,
            candidate_state,
        )
        layout = "packed_varlen"

    if args.sequences == 1:
        reference_out, reference_state = fp32_staged(
            legacy.chunk_gated_delta_rule_fwd_legacy,
            g32,
            beta32,
            scale,
            state32.clone(),
        )
        candidate_out, candidate_final_state = native_qkv(
            legacy.chunk_gated_delta_rule_fwd_legacy,
            g32,
            beta32,
            scale,
            state32.clone(),
        )
    else:
        reference_out, reference_state = fp32_staged(
            legacy.chunk_gated_delta_rule_fwd_legacy_varlen,
            g32,
            beta32,
            offsets,
            scale,
            state32.clone(),
        )
        candidate_out, candidate_final_state = native_qkv(
            legacy.chunk_gated_delta_rule_fwd_legacy_varlen,
            g32,
            beta32,
            offsets,
            scale,
            state32.clone(),
        )
    torch.cuda.synchronize()
    output_equal = torch.equal(reference_out, candidate_out)
    state_equal = torch.equal(reference_state, candidate_final_state)
    control_times = timed_ms(control, args.warmup, args.repeats)
    candidate_times = timed_ms(candidate, args.warmup, args.repeats)
    control_stats = stats(control_times)
    candidate_stats = stats(candidate_times)
    speedup = control_stats["median_ms"] / candidate_stats["median_ms"]
    report = {
        "device": torch.cuda.get_device_name(device),
        "layout": layout,
        "tokens": args.tokens,
        "sequences": args.sequences,
        "output_equal_after_fp32_cast": output_equal,
        "state_equal": state_equal,
        "control_fp32_qkv": control_stats,
        "candidate_native_qkv": candidate_stats,
        "median_speedup": speedup,
        "accept": (
            output_equal
            and state_equal
            and torch.isfinite(candidate_out).all().item()
            and torch.isfinite(candidate_final_state).all().item()
            and speedup >= 1.03
            and candidate_stats["p90_ms"] <= control_stats["p90_ms"]
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accept"]:
        raise SystemExit(
            "candidate did not meet equality and no-regression acceptance criteria"
        )


if __name__ == "__main__":
    main()
