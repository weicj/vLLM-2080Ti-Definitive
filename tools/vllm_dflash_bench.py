#!/usr/bin/env python3
import argparse
import dataclasses
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_synthetic_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    unit = (
        "record: vllm dflash versus mtp benchmark; synthetic filler; "
        "measure prefill, decode, acceptance, and wall time.\n"
    )
    prompt = "Read the records and continue the filler.\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    while len(ids) < target_tokens:
        prompt += unit
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_tokens:
        prompt = tokenizer.decode(ids[:target_tokens], skip_special_tokens=False)
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def build_quality_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    prefix = "Long filler text follows. FILLER START\n"
    suffix = "\nFILLER END\nReply with exactly: PROFILE_OK"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    filler_tokens = max(1, target_tokens - len(prefix_ids) - len(suffix_ids))
    prompt = prefix + (" the" * filler_tokens) + suffix
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def metrics_dict(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    if dataclasses.is_dataclass(metrics):
        return dataclasses.asdict(metrics)
    out: dict[str, Any] = {}
    for key in dir(metrics):
        if key.startswith("_"):
            continue
        value = getattr(metrics, key)
        if isinstance(value, (int, float, str, bool, type(None))):
            out[key] = value
    return out


def gpu_snapshot() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def build_llm_kwargs(args: argparse.Namespace, tokenizer_path: str, max_model_len: int, max_num_batched_tokens: int) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }
    if args.quantization != "auto":
        llm_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.block_size is not None:
        llm_kwargs["block_size"] = args.block_size
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.gdn_prefill_backend:
        llm_kwargs["gdn_prefill_backend"] = args.gdn_prefill_backend
    if args.language_model_only:
        llm_kwargs["language_model_only"] = True
    if args.skip_mm_profiling:
        llm_kwargs["skip_mm_profiling"] = True
    compilation_config = build_compilation_config(args)
    if compilation_config is not None:
        llm_kwargs["compilation_config"] = compilation_config
    return llm_kwargs


def build_speculative_config(args: argparse.Namespace, max_model_len: int) -> dict[str, Any] | None:
    if args.mode == "none":
        return None
    if args.mode == "mtp":
        return {
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    if args.mode == "dflash":
        if not args.draft:
            raise ValueError("--draft is required when --mode=dflash")
        config: dict[str, Any] = {
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.speculative_max_model_len or max_model_len,
            **({"attention_backend": args.speculative_attention_backend}
               if args.speculative_attention_backend else {}),
        }
        if args.draft_quantization != "none":
            config["quantization"] = args.draft_quantization
        if args.draft_tensor_parallel_size is not None:
            config["draft_tensor_parallel_size"] = args.draft_tensor_parallel_size
        if args.disable_padded_drafter_batch:
            config["disable_padded_drafter_batch"] = True
        if args.use_local_argmax_reduction:
            config["use_local_argmax_reduction"] = True
        return config
    raise ValueError(f"unsupported mode: {args.mode}")


def build_compilation_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.compilation_config_json:
        parsed = json.loads(args.compilation_config_json)
        if not isinstance(parsed, dict):
            raise ValueError("--compilation-config-json must decode to a JSON object")
        return parsed
    if args.mode == "none":
        return None
    capture = max(1, args.num_speculative_tokens + 1)
    return {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [capture],
        "max_cudagraph_capture_size": capture,
    }


def run_generate(
    llm: LLM,
    prompt: str,
    prompt_tokens: int,
    sampling: SamplingParams,
    *,
    label: str,
    task: str,
) -> dict[str, Any]:
    t0 = time.time()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    wall_s = time.time() - t0

    request = outputs[0]
    completion = request.outputs[0]
    text = completion.text or ""
    completion_tokens = len(completion.token_ids or [])
    metrics = metrics_dict(getattr(request, "metrics", None))

    ttft = None
    decode_s = None
    prefill_tps = None
    decode_tps = None
    first_token_ts = metrics.get("first_token_ts")
    scheduled_ts = metrics.get("scheduled_ts")
    last_token_ts = metrics.get("last_token_ts")
    if isinstance(first_token_ts, (int, float)) and isinstance(scheduled_ts, (int, float)):
        ttft = first_token_ts - scheduled_ts
        if ttft and ttft > 0:
            prefill_tps = prompt_tokens / ttft
    if isinstance(last_token_ts, (int, float)) and isinstance(first_token_ts, (int, float)):
        decode_s = last_token_ts - first_token_ts
        if decode_s and decode_s > 0:
            decode_tps = completion_tokens / decode_s

    return {
        "label": label,
        "task": task,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "ttft_s": ttft,
        "prefill_tps": prefill_tps,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "completion_tps_wall": (completion_tokens / wall_s) if wall_s > 0 else None,
        "content_sample": text[:300],
        "quality_ok": "PROFILE_OK" in text if task == "quality" else None,
        "metrics": metrics,
    }


def summarize_runs(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [run[key] for run in runs if isinstance(run.get(key), (int, float))]
    if not values:
        return None
    return statistics.mean(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft")
    parser.add_argument("--tokenizer")
    parser.add_argument("--mode", choices=("none", "mtp", "dflash"), required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--quality-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--quality-max-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--draft-quantization", default="none")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--draft-tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--speculative-max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--attention-backend")
    parser.add_argument("--speculative-attention-backend")
    parser.add_argument("--gdn-prefill-backend", choices=["flashinfer", "triton"])
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--disable-padded-drafter-batch", action="store_true")
    parser.add_argument("--use-local-argmax-reduction", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compilation-config-json")
    parser.add_argument("--warmup-max-tokens", type=int, default=8)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    synthetic_prompt, synthetic_prompt_tokens = build_synthetic_prompt(tokenizer, args.prompt_tokens)
    quality_prompt, quality_prompt_tokens = build_quality_prompt(tokenizer, args.quality_prompt_tokens)

    max_model_len = args.max_model_len or max(
        2048,
        synthetic_prompt_tokens + args.max_tokens + args.num_speculative_tokens + 128,
    )
    max_num_batched_tokens = args.max_num_batched_tokens or max(
        8192,
        synthetic_prompt_tokens + args.max_tokens + args.num_speculative_tokens + 256,
    )

    llm_kwargs = build_llm_kwargs(args, tokenizer_path, max_model_len, max_num_batched_tokens)
    speculative_config = build_speculative_config(args, max_model_len)
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config

    warmup_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=min(args.warmup_max_tokens, args.max_tokens),
        seed=args.seed,
        ignore_eos=True,
    )
    synthetic_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=True,
    )
    quality_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.quality_max_tokens,
        seed=args.seed,
        ignore_eos=False,
    )

    startup_t0 = time.time()
    llm = LLM(**llm_kwargs)
    startup_s = time.time() - startup_t0
    llm.generate(["Return OK."], warmup_sampling, use_tqdm=False)

    quality_run = None
    if not args.skip_quality:
        quality_run = run_generate(
            llm,
            quality_prompt,
            quality_prompt_tokens,
            quality_sampling,
            label=f"{args.case}-quality",
            task="quality",
        )

    synthetic_runs = []
    for idx in range(args.measured_runs):
        synthetic_runs.append(
            run_generate(
                llm,
                synthetic_prompt,
                synthetic_prompt_tokens,
                synthetic_sampling,
                label=f"{args.case}-synthetic-{idx + 1}",
                task="synthetic",
            )
        )

    if hasattr(llm, "llm_engine"):
        try:
            llm.llm_engine.do_log_stats()
        except Exception:
            pass

    result = {
        "case": args.case,
        "mode": args.mode,
        "model": args.model,
        "draft": args.draft,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "draft_quantization": args.draft_quantization if args.mode == "dflash" else None,
        "num_speculative_tokens": args.num_speculative_tokens if args.mode != "none" else 0,
        "startup_s": startup_s,
        "gpu_snapshot": gpu_snapshot(),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "llm_kwargs": llm_kwargs,
        "quality": quality_run,
        "synthetic_runs": synthetic_runs,
        "synthetic_summary": {
            "measured_runs": args.measured_runs,
            "prefill_tps_mean": summarize_runs(synthetic_runs, "prefill_tps"),
            "decode_tps_mean": summarize_runs(synthetic_runs, "decode_tps"),
            "wall_tps_mean": summarize_runs(synthetic_runs, "completion_tps_wall"),
            "wall_s_mean": summarize_runs(synthetic_runs, "wall_s"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
