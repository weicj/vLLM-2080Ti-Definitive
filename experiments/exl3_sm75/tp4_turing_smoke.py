"""Isolated four-T10 EXL3 TP4 quality smoke using the validated Turing runtime.

This intentionally exercises ExLlamaV3's default MoE expert-parallel plan:
dense modules are tensor-parallel while each routed expert stays whole.  It is
the semantic reference for the vLLM EXL3 bridge experiment.
"""

from __future__ import annotations

import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = os.environ.get(
    "BENCH_MODEL_DIR",
    "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4",
)
USE_PER_DEVICE = [
    float(value)
    for value in os.environ.get("BENCH_USE_PER_DEVICE", "14,14,14,14").split(",")
]
MAX_NEW_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "32"))
MAX_CONTEXT = int(os.environ.get("BENCH_CACHE_TOKENS", "512"))


def prompt_ids(tokenizer: Tokenizer) -> torch.Tensor:
    return tokenizer.hf_chat_template(
        [
            {
                "role": "user",
                "content": "Answer only the result: what is 17 plus 25?",
            }
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_job(generator: Generator, input_ids: torch.Tensor, stops: list[int]) -> dict:
    generator.enqueue(
        Job(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS + 1,
            sampler=GreedySampler(),
            stop_conditions=stops,
        )
    )
    text = ""
    token_ids: list[int] = []
    finished = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            text += result.get("text", "")
            ids = result.get("token_ids")
            if ids is not None:
                token_ids.extend(ids.reshape(-1).cpu().tolist())
            if result.get("eos"):
                finished = result
    if finished is None:
        raise RuntimeError("generation completed without EOS")
    return {
        "text": text,
        "token_ids": token_ids,
        "generated_tokens": finished["new_tokens"],
        "eos_reason": finished.get("eos_reason"),
        "prefill_seconds": finished["time_prefill"],
        "decode_seconds": finished["time_generate"],
    }


def main() -> None:
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for PLE SSD streaming")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected four visible GPUs, found {torch.cuda.device_count()}")
    if len(USE_PER_DEVICE) != 4:
        raise ValueError("BENCH_USE_PER_DEVICE must provide exactly four values")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=MAX_CONTEXT, max_batch_size=1)
    started = time.monotonic()
    try:
        model.load(
            tensor_p=True,
            tp_backend="nccl",
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=128,
            max_batch_size=1,
            verbose=True,
        )
        tokenizer = Tokenizer.from_config(config)
        ids = prompt_ids(tokenizer)
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            max_batch_size=1,
            max_chunk_size=128,
        )
        result = run_job(generator, ids, list(tokenizer.config.eos_token_id_list))
        memory = [
            {
                "device": index,
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(index),
            }
            for index in range(4)
        ]
        print(
            "TP4_TURING_SMOKE_JSON="
            + json.dumps(
                {
                    "load_seconds": time.monotonic() - started,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tp_plan": model.plan,
                    "prompt_tokens": ids.shape[1],
                    "result": result,
                    "memory": memory,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        model.unload()


if __name__ == "__main__":
    main()
