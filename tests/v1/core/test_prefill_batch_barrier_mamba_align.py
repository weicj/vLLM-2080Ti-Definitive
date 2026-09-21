# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill batch barrier must not livelock with mamba cache mode "align".

The barrier hands each peer `token_budget // cohort_width` tokens. With the
align split a step below one block quantum is clipped back to the chunk start,
so every peer ends up with 0 scheduled tokens and the frontier repeats forever.
"""

import os

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256

pytestmark = pytest.mark.cpu_test

# The suite's tiny text-only checkpoint needs an HF download; offline
# workstations can point this at a local checkpoint instead.
MODEL = os.environ.get("VLLM_SCHED_TEST_MODEL", "facebook/opt-125m")
_LOCAL_MODEL = os.path.isdir(MODEL)

MAMBA_BLOCK = 1600
NUM_SPEC = 1  # MTP on this checkpoint derives num_prefill_lookahead == 1
ATTN_BLOCK = 1600
MAX_MODEL_LEN = 36864
PROMPT_LENS = (4096, 8192, 32768)


def _build_scheduler(
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 2,
    long_prefill_token_threshold: int = 0,
) -> Scheduler:
    model_config = ModelConfig(
        model=MODEL,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=not _LOCAL_MODEL,
        max_model_len=MAX_MODEL_LEN,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=long_prefill_token_threshold,
        watermark=0.0,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=MAMBA_BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
        additional_config={"prefill_batch_barrier": True},
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4096,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=MAMBA_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=MAMBA_BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.num_prefill_lookahead = NUM_SPEC
    scheduler.use_eagle = True
    scheduler.use_eagle_block_drop = True
    return scheduler


def _request(request_id: str, prompt_len: int) -> Request:
    """Distinct prompts: an identical one resumes from the prefix cache."""
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(ATTN_BLOCK, sha256)
    sampling_params = SamplingParams(max_tokens=16, ignore_eos=True)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    token = 1 if request_id.endswith("0") else 2
    return Request(
        request_id=request_id,
        prompt_token_ids=[token] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
    )


def _model_output(scheduler: Scheduler, output) -> ModelRunnerOutput:
    # schedule() already advanced num_computed_tokens via _update_after_schedule,
    # so a chunk that reaches the prompt end is the one that samples.
    req_ids, sampled_token_ids = [], []
    for req_id in output.num_scheduled_tokens:
        request = scheduler.requests[req_id]
        req_ids.append(req_id)
        sampled_token_ids.append(
            [0]
            if request.num_computed_tokens
            >= request.num_tokens + request.num_output_placeholders
            else []
        )
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _drive(scheduler: Scheduler, requests: list[Request], max_steps: int = 400):
    """Return (steps, longest streak of empty steps while requests are running)."""
    empty_streak = longest_empty = 0
    for step in range(max_steps):
        output = scheduler.schedule()
        if output.num_scheduled_tokens:
            empty_streak = 0
            scheduler.update_from_output(output, _model_output(scheduler, output))
        elif scheduler.running:
            empty_streak += 1
            longest_empty = max(longest_empty, empty_streak)
            if empty_streak >= 2:
                return step + 1, longest_empty
        if all(request.is_finished() for request in requests):
            return step + 1, longest_empty
    return max_steps, longest_empty


def test_prefill_batch_barrier_mamba_align_never_stalls():
    for prompt_len in PROMPT_LENS:
        scheduler = _build_scheduler()
        requests = [_request("r0", prompt_len), _request("r1", prompt_len)]
        for request in requests:
            scheduler.add_request(request)
        steps, longest_empty = _drive(scheduler, requests)
        assert longest_empty < 2, (
            f"prompt_len={prompt_len}: {longest_empty} consecutive empty steps "
            "while both requests are running"
        )
        assert all(request.is_finished() for request in requests), (
            f"prompt_len={prompt_len}: requests unfinished after {steps} steps"
        )


def test_prefill_batch_barrier_mamba_align_survives_equal_frontier():
    """The live state: both peers at computed=3200 of 32768, step 2048//2."""
    scheduler = _build_scheduler()
    requests = [_request("r0", 32768), _request("r1", 32768)]
    scheduler.add_request(requests[0])
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output(scheduler, output))
    scheduler.add_request(requests[1])
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output(scheduler, output))
    assert [request.num_computed_tokens for request in requests] == [3200, 3200]

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"r0": 1024, "r1": 1024}


def test_prefill_batch_barrier_mamba_align_finishes_equal_tails_together():
    """A peer must not finish prefill by consuming its cohort's budget share."""
    scheduler = _build_scheduler()
    requests = [_request("r0", 4700), _request("r1", 4700)]
    scheduler.add_request(requests[0])
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output(scheduler, output))
    scheduler.add_request(requests[1])
    for _ in range(2):
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output(scheduler, output))
    assert [request.num_computed_tokens for request in requests] == [3200, 3200]

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"r0": 1024, "r1": 1024}
    scheduler.update_from_output(output, _model_output(scheduler, output))
    assert all(request.is_prefill_chunk for request in requests)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"r0": 476, "r1": 476}
    scheduler.update_from_output(output, _model_output(scheduler, output))
    assert all(not request.is_prefill_chunk for request in requests)


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "long_prefill_token_threshold", "max_num_seqs"),
    [
        (1024, 0, 2),
        (2048, 512, 2),
        (2048, 0, 3),
    ],
)
def test_prefill_batch_barrier_mamba_align_budget_edges_make_progress(
    max_num_batched_tokens: int,
    long_prefill_token_threshold: int,
    max_num_seqs: int,
):
    scheduler = _build_scheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        long_prefill_token_threshold=long_prefill_token_threshold,
    )
    requests = [_request(f"r{i}", 8192) for i in range(max_num_seqs)]
    for request in requests:
        scheduler.add_request(request)

    steps, longest_empty = _drive(scheduler, requests, max_steps=800)
    assert longest_empty < 2
    assert all(request.is_finished() for request in requests), (
        f"requests unfinished after {steps} steps"
    )
