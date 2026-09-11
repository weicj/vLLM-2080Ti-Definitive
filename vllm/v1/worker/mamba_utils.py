# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


@triton.jit
def postprocess_mamba_fused_kernel(
    num_accepted_tokens_ptr,
    mamba_state_idx_ptr,
    num_scheduled_tokens_ptr,
    num_computed_tokens_ptr,
    num_draft_tokens_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req: tl.int64,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    num_accepted_tokens_out_ptr,
    num_reqs,
    block_size: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    state_idx = tl.program_id(1)

    if req_idx >= num_reqs:
        return

    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)
    num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_idx)
    num_draft = tl.load(num_draft_tokens_ptr + req_idx)

    num_tokens_running_state = num_computed + num_scheduled - num_draft
    new_num_computed = num_tokens_running_state + num_accepted - 1
    aligned_new_computed = (new_num_computed // block_size) * block_size
    if aligned_new_computed < num_tokens_running_state:
        return

    accept_token_bias = aligned_new_computed - num_tokens_running_state
    dest_block_idx = aligned_new_computed // block_size - 1

    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)

    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + req_idx * block_table_stride_req

    src_block_id = tl.load(block_table_base + src_block_idx).to(tl.int64)
    dest_block_id = tl.load(block_table_base + dest_block_idx).to(tl.int64)
    is_conv_state = conv_width > 0

    if is_conv_state:
        src_offset = accept_token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        num_elems_to_copy = (conv_width - accept_token_bias).to(
            tl.int64
        ) * state_inner_size
        copy_size = num_elems_to_copy * state_elem_size
    else:
        actual_src_block_idx = src_block_idx + accept_token_bias
        actual_src_block_id = tl.load(block_table_base + actual_src_block_idx).to(
            tl.int64
        )
        src_addr = state_base_addr + actual_src_block_id * state_block_stride
        dst_addr = state_base_addr + dest_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size

    if src_block_idx == dest_block_idx and state_idx == 0:
        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)

    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = (i + offsets) < copy_size
        curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(curr_src, mask=mask)
        tl.store(curr_dst, data, mask=mask)


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


@dataclasses.dataclass
class MambaSpecDecodeGPUContext:
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    state_elem_sizes: torch.Tensor
    state_inner_sizes: torch.Tensor
    state_conv_widths: torch.Tensor
    state_group_indices: torch.Tensor
    block_size: int
    num_layers: int
    num_state_types: int
    mamba_group_ids: list[int]
    num_groups: int
    num_accepted_tokens_out: torch.Tensor
    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0
    mamba_state_idx_buf: CpuGpuBuffer | None = None
    num_scheduled_tokens_buf: CpuGpuBuffer | None = None
    num_computed_tokens_buf: CpuGpuBuffer | None = None
    num_draft_tokens_buf: CpuGpuBuffer | None = None
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        num_state_types: int,
        device: torch.device,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeGPUContext":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
        num_layers = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        )
        total_states = num_layers * num_state_types
        return cls(
            state_base_addrs=torch.zeros(total_states, dtype=torch.int64, device=device),
            state_block_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_inner_sizes=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            state_group_indices=torch.zeros(
                total_states, dtype=torch.int32, device=device
            ),
            block_size=mamba_spec.block_size,
            num_layers=num_layers,
            num_state_types=num_state_types,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            ),
            block_table_ptrs=torch.zeros(
                len(mamba_group_ids), dtype=torch.int64, device=device
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            is_initialized=False,
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        block_tables: list[torch.Tensor],
    ) -> None:
        if self.is_initialized:
            return

        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                for state_type_idx, state in enumerate(kv_caches):
                    self.state_base_addrs[idx] = state.data_ptr()
                    block_stride_elems = state.stride(0) if state.dim() > 1 else state.numel()
                    self.state_block_strides[idx] = (
                        block_stride_elems * state.element_size()
                    )
                    self.state_elem_sizes[idx] = state.element_size()

                    copy_func = mamba_state_copy_funcs[state_type_idx]
                    assert (
                        copy_func is get_conv_copy_spec
                        or copy_func is get_temporal_copy_spec
                    ), f"unexpected copy func: {copy_func}"
                    if copy_func is get_conv_copy_spec:
                        self.state_conv_widths[idx] = state.size(1) if state.dim() > 1 else 0
                        self.state_inner_sizes[idx] = (
                            state.stride(1) if state.dim() > 2 else 1
                        )
                    else:
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = (
                            state[0].numel() if state.dim() > 1 else 1
                        )

                    self.state_group_indices[idx] = group_local_idx
                    idx += 1

        assert len(block_tables) == self.num_groups, (
            f"expected {self.num_groups} block tables, got {len(block_tables)}"
        )
        strides = {bt.stride(0) for bt in block_tables}
        assert len(strides) == 1, (
            f"all mamba block tables must share stride(0), got {strides}"
        )
        self.block_table_stride_req = int(next(iter(strides)))
        for i, bt in enumerate(block_tables):
            self.block_table_ptrs[i] = bt.data_ptr()

        self.is_initialized = True

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
    ) -> None:
        if num_reqs == 0 or not self.is_initialized:
            return

        self.num_accepted_tokens_out[:num_reqs].copy_(
            num_accepted_tokens_gpu[:num_reqs]
        )

        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            mamba_state_idx_gpu,
            num_scheduled_tokens_gpu,
            num_computed_tokens_gpu,
            num_draft_tokens_gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.num_accepted_tokens_out,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
        )


@dataclasses.dataclass
class MambaBuffers:
    preprocess: MambaCopyBuffers
    postprocess_align: MambaSpecDecodeGPUContext | None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
        device: torch.device,
        with_postprocess_align: bool,
    ) -> "MambaBuffers":
        return cls(
            preprocess=MambaCopyBuffers.create(
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            postprocess_align=(
                MambaSpecDecodeGPUContext.create(
                    max_num_reqs=max_num_reqs,
                    kv_cache_config=kv_cache_config,
                    num_state_types=len(copy_funcs),
                    device=device,
                    make_buffer=make_buffer,
                )
                if with_postprocess_align
                else None
            ),
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def cleanup_mamba_state_idx(
    scheduler_output: SchedulerOutput,
    mamba_state_idx: dict[str, int],
) -> None:
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def postprocess_mamba_align_gpu(
    *,
    bufs: MambaBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [
                input_batch.block_table[gid].get_device_tensor(num_reqs)
                for gid in ctx.mamba_group_ids
            ],
        )

    ctx.run_fused_postprocess(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
        ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True
    )


def stage_postprocess_metadata_to_gpu(
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    num_scheduled_tokens_buf: CpuGpuBuffer,
    num_computed_tokens_buf: CpuGpuBuffer,
    num_draft_tokens_buf: CpuGpuBuffer,
) -> None:
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_scheduled = scheduler_output.num_scheduled_tokens
    scheduled_np = num_scheduled_tokens_buf.np
    computed_np = num_computed_tokens_buf.np
    draft_np = num_draft_tokens_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        scheduled_np[i] = num_scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, []))
    num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    num_computed_tokens_buf.copy_to_gpu(num_reqs)
    num_draft_tokens_buf.copy_to_gpu(num_reqs)


def stage_mamba_state_idx_to_gpu(
    mamba_state_idx: dict[str, int],
    req_ids: list[str],
    num_reqs: int,
    gpu_buf: CpuGpuBuffer,
) -> None:
    np_view = gpu_buf.np
    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before stage_mamba_state_idx_to_gpu"
        )
        np_view[i] = state_idx
    gpu_buf.copy_to_gpu(num_reqs)


def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None
    stage_mamba_state_idx_to_gpu(
        mamba_state_idx,
        req_ids,
        num_reqs,
        ctx.mamba_state_idx_buf,
    )
    stage_postprocess_metadata_to_gpu(
        scheduler_output,
        req_ids,
        num_reqs,
        requests,
        ctx.num_scheduled_tokens_buf,
        ctx.num_computed_tokens_buf,
        ctx.num_draft_tokens_buf,
    )
