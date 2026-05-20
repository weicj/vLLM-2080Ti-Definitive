# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionState,
    AttentionStatePair,
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup


class SpeculatorCudaGraphManagerBase(CudaGraphManager):
    """Base CudaGraphManager for draft models with a dedicated graph pool."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
    ):
        super().__init__(vllm_config, device, cudagraph_mode, decode_query_len)

        # Keep draft-model graph allocations isolated from the target model.
        if cudagraph_mode:
            self.pool = torch.cuda.graph_pool_handle()


class PrefillSpeculatorCudaGraphManager(SpeculatorCudaGraphManagerBase):
    """CudaGraphManager for draft prefill, using pre-built attention states
    from the target model's capture."""

    def capture(
        self,
        forward_fn: Callable,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> tuple[Callable[[CUDAGraphMode], None], AttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state_pair = attn_states[desc]
            attn_state = attn_state_pair.warmup if warmup else attn_state_pair.captured
            attn_metadata, slot_mappings = attn_state
            fwd = lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
            return fwd, attn_state

        super().capture(create_forward_fn, progress_bar_desc)


class DecodeSpeculatorCudaGraphManager(SpeculatorCudaGraphManagerBase):
    """CudaGraphManager for draft decode, building its own attention metadata."""

    def capture(
        self,
        forward_fn: Callable,
        model_state: ModelState,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> tuple[Callable[[CUDAGraphMode], None], AttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
            )
            attn_metadata, slot_mappings = attn_state

            fwd = lambda cg_mode: forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
            return fwd, attn_state

        super().capture(create_forward_fn, progress_bar_desc)
