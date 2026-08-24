# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state_in_place,
)
from flashinfer.prefill import single_prefill_with_kv_cache_return_lse

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import async_tensor_h2d, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)

_INT8KV_FA_PREFILL = os.getenv("VLLM_INT8KV_FA_PREFILL", "0") == "1"
_INT8KV_FI_PREFILL_BACKEND = os.getenv("VLLM_INT8KV_FLASHINFER_PREFILL_BACKEND", "fa2")
_INT8KV_FA_RAGGED_PREFILL = (
    os.getenv("VLLM_INT8KV_FA_RAGGED_PREFILL", "1") == "1"
)
_INT8KV_FA_FIRST_CHUNK_DEQUANT = (
    os.getenv("VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT", "0") == "1"
)
_INT8KV_FA_CONTINUATION_DEQUANT = (
    os.getenv("VLLM_INT8KV_FA_CONTINUATION_DEQUANT", "0") == "1"
)
_INT8KV_FA_CONTINUATION_MAX_TOKENS = int(
    os.getenv("VLLM_INT8KV_FA_CONTINUATION_MAX_TOKENS", "65536")
)
_INT8KV_FA_CONTINUATION_MIN_Q = int(
    os.getenv("VLLM_INT8KV_FA_CONTINUATION_MIN_Q", "128")
)
_INT8KV_FA_CASCADE_DEQUANT = os.getenv(
    "VLLM_INT8KV_FA_CASCADE_DEQUANT", "0"
) == "1"
_INT8KV_FA_CASCADE_TILE_TOKENS = int(
    os.getenv("VLLM_INT8KV_FA_CASCADE_TILE_TOKENS", "65536")
)
_INT8KV_FA_DIRECT_PAGED = os.getenv("VLLM_INT8KV_FA_DIRECT_PAGED", "0") == "1"
_INT8KV_ALIGNED_HEAD_STRIDE = (
    os.getenv("VLLM_INT8KV_ALIGNED_HEAD_STRIDE", "0") == "1"
)
_INT8KV_FI_PREFILL_WORKSPACES: dict[tuple[str, str], torch.Tensor] = {}
_INT8KV_FI_PREFILL_WRAPPERS: dict[tuple[object, ...], BatchPrefillWithRaggedKVCacheWrapper] = {}
_INT8KV_FI_PAGED_WRAPPERS: dict[tuple[object, ...], BatchPrefillWithRaggedKVCacheWrapper] = {}
_INT8KV_FI_KV_WORKSPACES: dict[
    tuple[str, str, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
_INT8KV_FA_PREFILL_USED = 0
_INT8KV_FA_DIRECT_USED = 0
_INT8KV_FA_PREFILL_SKIP_LOGGED: set[str] = set()
_INT8KV_FA_PREFILL_DEBUG_LOGGED = 0
_INT8KV_DEBUG_VERIFY = os.getenv("VLLM_INT8KV_DEBUG_VERIFY", "0") == "1"
_INT8KV_DEBUG_VERIFY_USED = 0
_INT8KV_DEBUG_COMPARE = os.getenv("VLLM_INT8KV_DEBUG_COMPARE", "0") == "1"
_INT8KV_DEBUG_COMPARE_USED = 0
_GEMMA4_SM75_SDPA_PREFILL512 = (
    os.getenv("VLLM_GEMMA4_SM75_SDPA_PREFILL512", "0") == "1"
)
_GEMMA4_SM75_SDPA_PREFILL512_GQA = (
    os.getenv("VLLM_GEMMA4_SM75_SDPA_PREFILL512_GQA", "0") == "1"
)
_GEMMA4_SM75_SDPA_PREFILL512_EXPAND = (
    os.getenv("VLLM_GEMMA4_SM75_SDPA_PREFILL512_EXPAND", "0") == "1"
)
_GEMMA4_SM75_SDPA_PREFILL512_USED = 0
_GEMMA4_SM75_FI_PREFILL256 = (
    os.getenv("VLLM_GEMMA4_SM75_FI_PREFILL256", "0") == "1"
)
_GEMMA4_SM75_FI_PREFILL256_BACKEND = os.getenv(
    "VLLM_GEMMA4_SM75_FI_PREFILL256_BACKEND",
    _INT8KV_FI_PREFILL_BACKEND,
)
_GEMMA4_SM75_FI_PREFILL256_USED = 0
_GEMMA4_FI_PREFILL_WRAPPERS: dict[
    tuple[object, ...], BatchPrefillWithRaggedKVCacheWrapper
] = {}

def _int8kv_direct_paged_jit_args(head_size: int) -> list[object]:
    variant_decl = r"""
#include <flashinfer/attention/variants.cuh>
namespace flashinfer {
DEFINE_HAS_MEMBER(maybe_k_scale_cache)
DEFINE_HAS_MEMBER(maybe_v_scale_cache)
DEFINE_HAS_MEMBER(paged_kv)

template <bool use_sliding_window, bool use_logits_soft_cap>
struct Int8TokenHeadScaleAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t window_left;
  float sm_scale_log2;
  float soft_cap_pre_tanh_scale;

  template <typename Params>
  __device__ __host__ Int8TokenHeadScaleAttention(const Params& params, uint32_t batch_idx,
                                                  uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    if constexpr (use_logits_soft_cap) {
      soft_cap_pre_tanh_scale = params.sm_scale * math::ptx_rcp(params.logits_soft_cap);
      sm_scale_log2 = math::log2e * params.logits_soft_cap;
    } else {
      sm_scale_log2 = params.sm_scale * math::log2e;
    }
  }

  template <typename Params>
  __device__ __forceinline__ uint32_t physical_scale_index(const Params& params,
                                                           uint32_t batch_idx,
                                                           uint32_t kv_idx,
                                                           uint32_t kv_head_idx) const {
    if constexpr (has_paged_kv_v<Params>) {
      uint32_t page_in_request;
      uint32_t slot;
      params.paged_kv.page_size.divmod(kv_idx, page_in_request, slot);
      uint32_t page_iter = params.paged_kv.indptr[batch_idx] + page_in_request;
      uint32_t physical_page = __ldg(params.paged_kv.indices + page_iter);
      return physical_page * params.scale_stride_page + slot * params.scale_stride_slot +
             kv_head_idx * params.scale_stride_head;
    } else {
      return kv_idx * params.scale_stride_slot + kv_head_idx * params.scale_stride_head;
    }
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float k_scale = params.maybe_k_scale_cache[physical_scale_index(params, batch_idx, kv_idx, kv_head_idx)];
    logits = logits * k_scale;
    if constexpr (use_logits_soft_cap) {
      logits = float(math::tanh(logits * soft_cap_pre_tanh_scale)) * params.logits_soft_cap;
    }
    return logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    bool mask = true;
    if constexpr (use_sliding_window) {
      mask &= (kv_idx + 1 + window_left >= kv_len);
    }
    return mask;
  })

  REGISTER_VALUE_TRANSFORM(params, value, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float v_scale = params.maybe_v_scale_cache[physical_scale_index(params, batch_idx, kv_idx, kv_head_idx)];
    return static_cast<T>(static_cast<float>(value) * v_scale);
  })

  REGISTER_PROBABILITY_TRANSFORM(params, prob, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    float v_scale = params.maybe_v_scale_cache[physical_scale_index(params, batch_idx, kv_idx, kv_head_idx)];
    return static_cast<T>(static_cast<float>(prob) * v_scale);
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, softmax_scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    return output * d_rcp;
  })
};
}
"""
    return [
        f"vllm_int8kv_tokenscale_paged_prefill_sm75_d{head_size}_v1",
        torch.float16,
        torch.int8,
        torch.float16,
        torch.int32,
        head_size,
        head_size,
        ["maybe_k_scale_cache", "maybe_v_scale_cache"],
        ["float", "float"],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "gqa_group_size",
            "scale_stride_page",
            "scale_stride_slot",
            "scale_stride_head",
        ],
        ["double", "double", "double", "double", "uint32_t", "uint32_t", "uint32_t", "uint32_t"],
        "Int8TokenHeadScaleAttention<false, false>",
        variant_decl,
    ]


def _int8kv_normalize_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_int8kv_flashinfer_prefill_workspace(
    device: torch.device,
    backend: str,
) -> torch.Tensor:
    device = _int8kv_normalize_cuda_device(device)
    key = (str(device), backend)
    workspace = _INT8KV_FI_PREFILL_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _INT8KV_FI_PREFILL_WORKSPACES[key] = workspace
    return workspace


def _get_int8kv_flashinfer_kv_workspace(
    device: torch.device,
    dtype: torch.dtype,
    num_kv_heads: int,
    head_size: int,
    min_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _int8kv_normalize_cuda_device(device)
    capacity = 1 << (int(min_tokens) - 1).bit_length()
    key = (str(device), str(dtype), num_kv_heads, head_size, capacity)
    workspaces = _INT8KV_FI_KV_WORKSPACES.get(key)
    if workspaces is None:
        k_workspace = torch.empty(
            (capacity, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        v_workspace = torch.empty_like(k_workspace)
        workspaces = (k_workspace, v_workspace)
        _INT8KV_FI_KV_WORKSPACES[key] = workspaces
    return workspaces


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_computed_tokens_cpu: torch.Tensor | None
    is_prefilling: torch.Tensor | None

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None

    @staticmethod
    def compute_mm_prefix_range_tensor(
        mm_prefix_range: dict[int, list[tuple[int, int]]] | None,
        num_seqs: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        if mm_prefix_range is None:
            return None

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Build on CPU first then move to GPU in a single H2D transfer
        max_ranges = max(len(r) for r in range_lists)
        # Pad all sequences to the same number of ranges
        padded = []
        for r in range_lists:
            padded_r = list(r) + [(0, 0)] * (max_ranges - len(r))
            padded.append(padded_r)
        # Build on pinned CPU memory so the H2D transfer is non-blocking.
        padded = async_tensor_h2d(padded, dtype=torch.int32, device=device)
        return padded.view(num_seqs, max_ranges, 2)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        try:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        except Exception:
            seq_lens_cpu = None
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        try:
            num_computed_tokens_cpu = common_attn_metadata.num_computed_tokens_cpu
        except Exception:
            num_computed_tokens_cpu = None

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            is_prefilling=common_attn_metadata.is_prefilling,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_tensor",  # [FORK-PORT] PR#41505
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            # Pad head_size by sizeof(float32)/sizeof(cache_dtype) so
            # the per-head scale fits inline.  The backend extracts
            # data[:head_size] and scale[head_size:] via typed views.
            from vllm.utils.torch_utils import (
                STR_DTYPE_TO_TORCH_DTYPE,
                get_dtype_size,
            )

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
            padded_head_size = head_size + scale_pad
            if (
                cache_dtype is torch.int8
                and _INT8KV_ALIGNED_HEAD_STRIDE
                and padded_head_size % 16 != 0
            ):
                padded_head_size = ((padded_head_size + 15) // 16) * 16
            return (num_blocks, 2, block_size, num_kv_heads, padded_head_size)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")
        return stride_order

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _debug_verify_per_token_head_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        global _INT8KV_DEBUG_VERIFY_USED
        if not _INT8KV_DEBUG_VERIFY or _INT8KV_DEBUG_VERIFY_USED >= 4:
            return
        if key.numel() == 0 or value.numel() == 0:
            return
        if self._k_scale_cache is None or self._v_scale_cache is None:
            return

        slots = slot_mapping.detach().cpu().tolist()
        if not slots:
            return

        key_ref = key.detach().float().cpu()
        value_ref = value.detach().float().cpu()
        key_cache_cpu = key_cache.detach().cpu().float()
        value_cache_cpu = value_cache.detach().cpu().float()
        k_scale_cpu = self._k_scale_cache.detach().cpu()
        v_scale_cpu = self._v_scale_cache.detach().cpu()

        k_err = 0.0
        v_err = 0.0
        for tok_idx, slot in enumerate(slots):
            if slot < 0:
                continue
            blk = slot // key_cache.shape[1]
            slot_in_blk = slot % key_cache.shape[1]
            k_deq = key_cache_cpu[blk, slot_in_blk, :, :self.head_size] * (
                k_scale_cpu[blk, slot_in_blk, :].unsqueeze(-1)
            )
            v_deq = value_cache_cpu[blk, slot_in_blk, :, :value.shape[-1]] * (
                v_scale_cpu[blk, slot_in_blk, :].unsqueeze(-1)
            )
            k_err = max(k_err, float((k_deq - key_ref[tok_idx]).abs().max()))
            v_err = max(v_err, float((v_deq - value_ref[tok_idx]).abs().max()))

        _INT8KV_DEBUG_VERIFY_USED += 1
        logger.info(
            "INT8KV verify count=%d key_contig=%s value_contig=%s "
            "key_stride=%s value_stride=%s k_err=%.6f v_err=%.6f",
            _INT8KV_DEBUG_VERIFY_USED,
            key.is_contiguous(),
            value.is_contiguous(),
            tuple(key.stride()),
            tuple(value.stride()),
            k_err,
            v_err,
        )

    def _debug_compare_single_token_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        output: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> None:
        global _INT8KV_DEBUG_COMPARE_USED
        if not _INT8KV_DEBUG_COMPARE or _INT8KV_DEBUG_COMPARE_USED >= 64:
            return
        if output.shape[0] != 1 or int(attn_metadata.seq_lens.shape[0]) != 1:
            return
        if self.sliding_window != (-1, -1):
            return
        if self._k_scale_cache is None or self._v_scale_cache is None:
            return

        seq_len = int(attn_metadata.seq_lens[0].item())
        block_size = int(key_cache.shape[1])
        num_blocks = (seq_len + block_size - 1) // block_size
        page_ids = attn_metadata.block_table[0, :num_blocks].to(dtype=torch.long)

        k_pages = key_cache.index_select(0, page_ids)[..., : self.head_size]
        v_pages = value_cache.index_select(0, page_ids)[..., : self.head_size]
        ks_pages = self._k_scale_cache.index_select(0, page_ids)
        vs_pages = self._v_scale_cache.index_select(0, page_ids)

        k_ref = (
            k_pages.reshape(num_blocks * block_size, self.num_kv_heads, self.head_size)[
                :seq_len
            ].float()
            * ks_pages.reshape(num_blocks * block_size, self.num_kv_heads)[:seq_len]
            .float()
            .unsqueeze(-1)
        )
        v_ref = (
            v_pages.reshape(num_blocks * block_size, self.num_kv_heads, self.head_size)[
                :seq_len
            ].float()
            * vs_pages.reshape(num_blocks * block_size, self.num_kv_heads)[:seq_len]
            .float()
            .unsqueeze(-1)
        )
        q_ref = query[0].float().unsqueeze(1)
        k_heads = k_ref.repeat_interleave(self.num_queries_per_kv, dim=1).permute(
            1, 0, 2
        )
        v_heads = v_ref.repeat_interleave(self.num_queries_per_kv, dim=1).permute(
            1, 0, 2
        )
        scores = (q_ref * k_heads).sum(-1) * self.scale
        probs = torch.softmax(scores, dim=-1)
        ref = torch.einsum("ht,thd->hd", probs, v_heads.permute(1, 0, 2)).to(
            output.dtype
        )
        err = (output[0] - ref).abs()
        _INT8KV_DEBUG_COMPARE_USED += 1
        logger.info(
            "INT8KV compare count=%d layer=%s seq_len=%d max_err=%.6f mean_err=%.6f "
            "out0=%s ref0=%s",
            _INT8KV_DEBUG_COMPARE_USED,
            getattr(layer, "layer_name", "<unknown>"),
            seq_len,
            float(err.max().item()),
            float(err.mean().item()),
            output[0, 0, :8].detach().cpu().tolist(),
            ref[0, :8].detach().cpu().tolist(),
        )

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded head dimension.

        The KV cache shape is ``(num_blocks, 2, block_size, nkv, hs+pad)``
        where ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The last
        ``pad`` elements of each head hold one float32 scale.  We create
        strided float32 views over those bytes.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, _, block_size, nkv, padded_hs = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        hs = self.head_size
        if hs + scale_pad > padded_hs:
            hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        # In the raw bytes, each (block, kv_half, slot, head) occupies
        # padded_hs * dtype_sz bytes.  The scale float32 sits at byte
        # offset hs * dtype_sz within that region.
        kv_half_bytes = block_size * nkv * padded_hs * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4  # stride between blocks
        slot_f32 = nkv * padded_hs * dtype_sz // 4  # stride between slots
        head_f32 = padded_hs * dtype_sz // 4  # stride between heads
        scale_off_f32 = hs * dtype_sz // 4  # offset to scale within head

        # K scales: kv_half=0
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales: kv_half=1, offset by kv_half_bytes
        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

    def _flashinfer_indptr(
        self,
        indptr: torch.Tensor,
        num_heads: int,
        head_dim: int,
        backend: str | None = None,
    ) -> torch.Tensor:
        if (backend or _INT8KV_FI_PREFILL_BACKEND) == "cudnn":
            return indptr * (num_heads * head_dim)
        return indptr

    def _get_or_plan_gemma4_flashinfer_prefill_wrapper(
        self,
        device: torch.device,
        plan_key: tuple[object, ...],
        plan_kwargs: dict[str, object],
    ) -> BatchPrefillWithRaggedKVCacheWrapper:
        norm_device = _int8kv_normalize_cuda_device(device)
        cache_key = (
            str(norm_device),
            _GEMMA4_SM75_FI_PREFILL256_BACKEND,
            *plan_key,
        )
        wrapper = _GEMMA4_FI_PREFILL_WRAPPERS.get(cache_key)
        if wrapper is None:
            workspace = _get_int8kv_flashinfer_prefill_workspace(
                norm_device, _GEMMA4_SM75_FI_PREFILL256_BACKEND
            )
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                "NHD",
                backend=_GEMMA4_SM75_FI_PREFILL256_BACKEND,
            )
            wrapper.plan(**plan_kwargs)
            _GEMMA4_FI_PREFILL_WRAPPERS[cache_key] = wrapper
        return wrapper

    def _get_or_plan_int8kv_flashinfer_prefill_wrapper(
        self,
        device: torch.device,
        plan_key: tuple[object, ...],
        plan_kwargs: dict[str, object],
    ) -> BatchPrefillWithRaggedKVCacheWrapper:
        norm_device = _int8kv_normalize_cuda_device(device)
        cache_key = (str(norm_device), _INT8KV_FI_PREFILL_BACKEND, *plan_key)
        wrapper = _INT8KV_FI_PREFILL_WRAPPERS.get(cache_key)
        if wrapper is None:
            workspace = _get_int8kv_flashinfer_prefill_workspace(
                norm_device, _INT8KV_FI_PREFILL_BACKEND
            )
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                "NHD",
                backend=_INT8KV_FI_PREFILL_BACKEND,
            )
            wrapper.plan(**plan_kwargs)
            _INT8KV_FI_PREFILL_WRAPPERS[cache_key] = wrapper
        return wrapper

    def _get_int8kv_flashinfer_paged_wrapper(
        self,
        device: torch.device,
        plan_key: tuple[object, ...],
    ) -> BatchPrefillWithPagedKVCacheWrapper:
        norm_device = _int8kv_normalize_cuda_device(device)
        cache_key = (str(norm_device), _INT8KV_FI_PREFILL_BACKEND, *plan_key)
        wrapper = _INT8KV_FI_PAGED_WRAPPERS.get(cache_key)
        if wrapper is None:
            workspace = _get_int8kv_flashinfer_prefill_workspace(
                norm_device, _INT8KV_FI_PREFILL_BACKEND
            )
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                backend=_INT8KV_FI_PREFILL_BACKEND,
                jit_args=_int8kv_direct_paged_jit_args(self.head_size),
            )
            _INT8KV_FI_PAGED_WRAPPERS[cache_key] = wrapper
        return wrapper

    def _try_int8kv_direct_paged_prefill(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        num_actual_tokens: int,
        q_seq_lens: torch.Tensor,
        seq_len: int,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> bool:
        if not _INT8KV_FA_DIRECT_PAGED:
            return False
        if output_scale is not None or output_block_scale is not None:
            return False
        if q_seq_lens.numel() != 1 or attn_metadata.seq_lens_cpu is None:
            return False

        global _INT8KV_FA_DIRECT_USED
        try:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
                return False
            key_cache = key_cache[..., : self.head_size]
            value_cache = value_cache[..., : self.head_size]
            block_size = int(key_cache.shape[1])
            q_len = int(q_seq_lens[0].item())
            if q_len <= 0 or seq_len < q_len:
                return False
            nblocks = (seq_len + block_size - 1) // block_size
            page_indices = attn_metadata.block_table[0, :nblocks].to(
                dtype=torch.int32, device=query.device
            )
            if _INT8KV_ALIGNED_HEAD_STRIDE:
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
            else:
                page_ids = page_indices.to(dtype=torch.long)
                key_cache = key_cache.index_select(0, page_ids).contiguous()
                value_cache = value_cache.index_select(0, page_ids).contiguous()
                k_scale_cache = self._k_scale_cache.index_select(0, page_ids).contiguous()
                v_scale_cache = self._v_scale_cache.index_select(0, page_ids).contiguous()
                page_indices = torch.arange(nblocks, dtype=torch.int32, device=query.device)
            last_page_len = seq_len - (nblocks - 1) * block_size
            qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=query.device)
            kv_indptr = torch.tensor(
                [0, nblocks], dtype=torch.int32, device=query.device
            )
            kv_last_page_len = torch.tensor(
                [last_page_len], dtype=torch.int32, device=query.device
            )
            q_prefill = query[:num_actual_tokens]
            if not q_prefill.is_contiguous():
                q_prefill = q_prefill.contiguous()
            out = output[:num_actual_tokens]
            plan_key = (
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                str(query.dtype),
                str(key_cache.dtype),
                q_len,
                seq_len,
                block_size,
            )
            wrapper = self._get_int8kv_flashinfer_paged_wrapper(
                query.device,
                plan_key,
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=page_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                page_size=block_size,
                causal=True,
                window_left=self.sliding_window[0],
                logits_soft_cap=self.logits_soft_cap,
                sm_scale=self.scale,
                pos_encoding_mode="NONE",
                q_data_type=query.dtype,
                kv_data_type=key_cache.dtype,
                o_data_type=output.dtype,
                seq_lens=torch.tensor(
                    [seq_len], dtype=torch.int32, device=query.device
                ),
                seq_lens_q=q_seq_lens.to(dtype=torch.int32, device=query.device),
                max_token_per_sequence=q_len,
            )
            wrapper.run(
                q_prefill,
                (key_cache, value_cache),
                k_scale_cache,
                v_scale_cache,
                0.0,
                self.scale,
                1.0,
                1e-4,
                self.num_queries_per_kv,
                k_scale_cache.stride(0),
                k_scale_cache.stride(1),
                k_scale_cache.stride(2),
                out=out,
            )
            _INT8KV_FA_DIRECT_USED += 1
            if _INT8KV_FA_DIRECT_USED <= 4 or _INT8KV_FA_DIRECT_USED % 64 == 0:
                logger.info(
                    "INT8 KV FlashInfer direct paged used count=%d backend=%s tokens=%d kv_tokens=%d q_len=%d heads=%d kv_heads=%d head_dim=%d",
                    _INT8KV_FA_DIRECT_USED,
                    _INT8KV_FI_PREFILL_BACKEND,
                    num_actual_tokens,
                    seq_len,
                    q_len,
                    query.shape[1],
                    self.num_kv_heads,
                    self.head_size,
                )
            return True
        except Exception as e:
            skip_reason = f"direct_paged_failed:{type(e).__name__}"
            if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                logger.info("INT8 KV FlashInfer prefill skipped reason=%s err=%s", skip_reason, e)
            return False

    def _dequantize_int8kv_cache_range(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = key_cache.shape[1]
        start_block = start // block_size
        end_block = (end + block_size - 1) // block_size
        offset = start - start_block * block_size
        num_tokens = end - start
        blocks = block_table[0, start_block:end_block].to(torch.long)
        k_workspace, v_workspace = _get_int8kv_flashinfer_kv_workspace(
            key_cache.device,
            dtype,
            self.num_kv_heads,
            self.head_size,
            num_tokens,
        )
        k_target = k_workspace[:num_tokens]
        v_target = v_workspace[:num_tokens]
        k_data = key_cache[blocks, :, :, : self.head_size].reshape(
            -1, self.num_kv_heads, self.head_size
        )[offset : offset + num_tokens]
        v_data = value_cache[blocks, :, :, : self.head_size].reshape(
            -1, self.num_kv_heads, self.head_size
        )[offset : offset + num_tokens]
        k_scale = self._k_scale_cache[blocks].reshape(
            -1, self.num_kv_heads
        )[offset : offset + num_tokens]
        v_scale = self._v_scale_cache[blocks].reshape(
            -1, self.num_kv_heads
        )[offset : offset + num_tokens]
        torch.mul(k_data.to(torch.float32), k_scale.unsqueeze(-1), out=k_target)
        torch.mul(v_data.to(torch.float32), v_scale.unsqueeze(-1), out=v_target)
        return k_target, v_target

    def _run_int8kv_cascade_flashinfer_prefill(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_len: int,
        q_len: int,
    ) -> torch.Tensor:
        prefix_len = seq_len - q_len
        tile_tokens = max(1, _INT8KV_FA_CASCADE_TILE_TOKENS)
        state_v = None
        state_s = None

        def merge_segment(start: int, end: int, causal: bool) -> None:
            nonlocal state_v, state_s
            k_tile, v_tile = self._dequantize_int8kv_cache_range(
                key_cache,
                value_cache,
                attn_metadata.block_table,
                start,
                end,
                query.dtype,
            )
            seg_v, seg_s = single_prefill_with_kv_cache_return_lse(
                query,
                k_tile,
                v_tile,
                causal=causal,
                kv_layout="NHD",
                sm_scale=self.scale,
                window_left=self.sliding_window[0],
                logits_soft_cap=self.logits_soft_cap,
                backend=_INT8KV_FI_PREFILL_BACKEND,
                return_lse=True,
            )
            if state_v is None:
                state_v = seg_v
                state_s = seg_s
            else:
                merge_state_in_place(state_v, state_s, seg_v, seg_s)

        tile_start = 0
        while tile_start < prefix_len:
            tile_end = min(tile_start + tile_tokens, prefix_len)
            merge_segment(tile_start, tile_end, False)
            tile_start = tile_end
        merge_segment(prefix_len, seq_len, True)
        return state_v

    def _store_int8kv_fa_prefill_output(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if output_scale is None:
            if output_block_scale is not None:
                return False
            output.copy_(attn_output)
            return True
        if output_block_scale is not None:
            return False

        try:
            dtype_info = torch.finfo(output.dtype)
        except TypeError:
            return False
        scaled = attn_output.mul(torch.reciprocal(output_scale))
        scaled.clamp_(dtype_info.min, dtype_info.max)
        output.copy_(scaled)
        return True

    def _try_int8kv_fa_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        num_actual_tokens: int,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if not _INT8KV_FA_PREFILL or not self._is_per_token_head_quant:
            return False

        global _INT8KV_FA_PREFILL_USED, _INT8KV_FA_PREFILL_DEBUG_LOGGED
        skip_reason = None
        q_lens_cpu = attn_metadata.query_start_loc_cpu
        q_lens_cpu = q_lens_cpu[1:] - q_lens_cpu[:-1]
        num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu
        if (
            os.getenv("VLLM_INT8KV_FA_PREFILL_DEBUG", "0") == "1"
            and _INT8KV_FA_PREFILL_DEBUG_LOGGED < 16
        ):
            _INT8KV_FA_PREFILL_DEBUG_LOGGED += 1
            try:
                logger.info(
                    "INT8 KV FlashInfer prefill debug count=%d actual=%d max_q=%d max_seq=%d q_lens=%s seq_lens=%s computed=%s",
                    _INT8KV_FA_PREFILL_DEBUG_LOGGED,
                    num_actual_tokens,
                    attn_metadata.max_query_len,
                    attn_metadata.max_seq_len,
                    q_lens_cpu.tolist(),
                    None
                    if attn_metadata.seq_lens_cpu is None
                    else attn_metadata.seq_lens_cpu.tolist(),
                    None
                    if num_computed_tokens_cpu is None
                    else num_computed_tokens_cpu.tolist(),
                )
            except Exception as e:
                logger.info("INT8 KV FlashInfer prefill debug failed: %s", e)
        if self.kv_cache_dtype != "int8_per_token_head":
            skip_reason = "not_int8_per_token_head"
        elif self.attn_type != AttentionType.DECODER:
            skip_reason = "not_decoder_attention"
        elif self.alibi_slopes is not None:
            skip_reason = "alibi"
        elif self.sinks is not None:
            skip_reason = "attention_sinks"
        elif self.sliding_window != (-1, -1):
            skip_reason = "sliding_window"
        elif self.logits_soft_cap not in (None, 0.0):
            skip_reason = "logits_soft_cap"
        elif query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16:
            skip_reason = "non_fp16_qkv"
        elif num_computed_tokens_cpu is None:
            skip_reason = "missing_num_computed_tokens"

        if skip_reason is not None:
            if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                logger.info("INT8 KV FlashInfer prefill skipped reason=%s", skip_reason)
            return False

        indptr_cpu = attn_metadata.query_start_loc_cpu.to(torch.int32)
        q_seq_lens = indptr_cpu[1:] - indptr_cpu[:-1]
        is_first_chunk = bool((num_computed_tokens_cpu == 0).all().item())
        use_continuation_bridge = False
        use_cascade_bridge = False
        force_first_chunk_dequant = False
        seq_len = int(attn_metadata.max_seq_len)
        if is_first_chunk and _INT8KV_FA_FIRST_CHUNK_DEQUANT:
            if q_seq_lens.numel() != 1:
                skip_reason = "first_chunk_batch_not_1"
            elif attn_metadata.seq_lens_cpu is not None:
                seq_len = int(attn_metadata.seq_lens_cpu[0].item())
                use_continuation_bridge = True
                force_first_chunk_dequant = True
            else:
                use_continuation_bridge = True
                force_first_chunk_dequant = True
        elif not is_first_chunk:
            if not _INT8KV_FA_CONTINUATION_DEQUANT:
                skip_reason = "prefix_or_cached_kv"
            elif q_seq_lens.numel() != 1:
                skip_reason = "continuation_batch_not_1"
            elif int(q_seq_lens[0].item()) < _INT8KV_FA_CONTINUATION_MIN_Q:
                skip_reason = "continuation_q_too_small"
            elif attn_metadata.seq_lens_cpu is None:
                skip_reason = "continuation_missing_seq_lens_cpu"
            else:
                seq_len = int(attn_metadata.seq_lens_cpu[0].item())
                if seq_len > _INT8KV_FA_CONTINUATION_MAX_TOKENS:
                    if _INT8KV_FA_CASCADE_DEQUANT:
                        use_cascade_bridge = True
                    else:
                        skip_reason = "continuation_too_long"
                else:
                    use_continuation_bridge = True
                if use_cascade_bridge:
                    use_continuation_bridge = True

        if skip_reason is not None:
            if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                logger.info("INT8 KV FlashInfer prefill skipped reason=%s", skip_reason)
            return False

        q_prefill = query[:num_actual_tokens]
        if not q_prefill.is_contiguous():
            q_prefill = q_prefill.contiguous()

        if not use_continuation_bridge:
            if self._try_int8kv_direct_paged_prefill(
                q_prefill,
                kv_cache,
                output,
                attn_metadata,
                num_actual_tokens,
                q_seq_lens,
                seq_len,
                output_scale,
                output_block_scale,
            ):
                return True

        if use_cascade_bridge:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if not force_first_chunk_dequant and self._try_int8kv_direct_paged_prefill(
                q_prefill,
                kv_cache,
                output,
                attn_metadata,
                num_actual_tokens,
                q_seq_lens,
                seq_len,
                output_scale,
                output_block_scale,
            ):
                return True
            out = self._run_int8kv_cascade_flashinfer_prefill(
                q_prefill,
                key_cache,
                value_cache,
                attn_metadata,
                seq_len,
                int(q_seq_lens[0].item()),
            )
            if not self._store_int8kv_fa_prefill_output(
                out,
                output[:num_actual_tokens],
                output_scale,
                output_block_scale,
            ):
                skip_reason = "unsupported_output_quant"
                if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                    _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                    logger.info(
                        "INT8 KV FlashInfer prefill skipped reason=%s",
                        skip_reason,
                    )
                return False
            _INT8KV_FA_PREFILL_USED += 1
            if _INT8KV_FA_PREFILL_USED <= 4 or _INT8KV_FA_PREFILL_USED % 64 == 0:
                logger.info(
                    "INT8 KV FlashInfer prefill used count=%d backend=%s tokens=%d kv_tokens=%d bridge=cascade tile=%d max_q_len=%d heads=%d kv_heads=%d head_dim=%d",
                    _INT8KV_FA_PREFILL_USED,
                    _INT8KV_FI_PREFILL_BACKEND,
                    num_actual_tokens,
                    seq_len,
                    _INT8KV_FA_CASCADE_TILE_TOKENS,
                    attn_metadata.max_query_len,
                    query.shape[1],
                    key.shape[1],
                    self.head_size,
                )
            return True

        kv_indptr_cpu = indptr_cpu
        max_sequence_kv = attn_metadata.max_seq_len
        if use_continuation_bridge:
            if not force_first_chunk_dequant and self._try_int8kv_direct_paged_prefill(
                q_prefill,
                kv_cache,
                output,
                attn_metadata,
                num_actual_tokens,
                q_seq_lens,
                seq_len,
                output_scale,
                output_block_scale,
            ):
                return True
            kv_indptr_cpu = torch.tensor(
                [0, seq_len],
                dtype=torch.int32,
                device=indptr_cpu.device,
            )
            max_sequence_kv = seq_len
        plan_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            str(query.dtype),
            str(key.dtype),
            tuple(int(x) for x in indptr_cpu.tolist()),
            tuple(int(x) for x in kv_indptr_cpu.tolist()),
            attn_metadata.max_query_len,
            max_sequence_kv,
        )
        wrapper = self._get_or_plan_int8kv_flashinfer_prefill_wrapper(
            query.device,
            plan_key,
            {
                "qo_indptr": self._flashinfer_indptr(
                    indptr_cpu, self.num_heads, self.head_size
                ),
                "kv_indptr": self._flashinfer_indptr(
                    kv_indptr_cpu, self.num_kv_heads, self.head_size
                ),
                "num_qo_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim_qk": self.head_size,
                "causal": True,
                "window_left": self.sliding_window[0],
                "logits_soft_cap": self.logits_soft_cap,
                "sm_scale": self.scale,
                "pos_encoding_mode": "NONE",
                "q_data_type": query.dtype,
                "kv_data_type": key.dtype,
                "seq_lens": kv_indptr_cpu[1:] - kv_indptr_cpu[:-1],
                "seq_lens_q": q_seq_lens,
                "max_token_per_sequence": attn_metadata.max_query_len,
                "max_sequence_kv": max_sequence_kv,
            },
        )
        if use_continuation_bridge:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            block_size = key_cache.shape[1]
            nblocks = (seq_len + block_size - 1) // block_size
            blocks = attn_metadata.block_table[0, :nblocks].to(torch.long)
            k_workspace, v_workspace = _get_int8kv_flashinfer_kv_workspace(
                query.device,
                query.dtype,
                self.num_kv_heads,
                self.head_size,
                seq_len,
            )
            k_target = k_workspace[:seq_len]
            v_target = v_workspace[:seq_len]
            k_data = key_cache[blocks, :, :, : self.head_size].reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            v_data = value_cache[blocks, :, :, : self.head_size].reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            k_scale = self._k_scale_cache[blocks].reshape(
                -1, self.num_kv_heads
            )[:seq_len]
            v_scale = self._v_scale_cache[blocks].reshape(
                -1, self.num_kv_heads
            )[:seq_len]
            torch.mul(k_data.to(torch.float32), k_scale.unsqueeze(-1), out=k_target)
            torch.mul(v_data.to(torch.float32), v_scale.unsqueeze(-1), out=v_target)
            k_prefill = k_target
            v_prefill = v_target
        else:
            k_prefill = key[:num_actual_tokens]
            v_prefill = value[:num_actual_tokens]
            if not k_prefill.is_contiguous():
                k_prefill = k_prefill.contiguous()
            if not v_prefill.is_contiguous():
                v_prefill = v_prefill.contiguous()
        if not _INT8KV_FA_RAGGED_PREFILL:
            skip_reason = "ragged_prefill_disabled"
            if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                logger.info("INT8 KV FlashInfer prefill skipped reason=%s", skip_reason)
            return False
        out = wrapper.run(q_prefill, k_prefill, v_prefill)
        if not self._store_int8kv_fa_prefill_output(
            out,
            output[:num_actual_tokens],
            output_scale,
            output_block_scale,
        ):
            skip_reason = "unsupported_output_quant"
            if skip_reason not in _INT8KV_FA_PREFILL_SKIP_LOGGED:
                _INT8KV_FA_PREFILL_SKIP_LOGGED.add(skip_reason)
                logger.info("INT8 KV FlashInfer prefill skipped reason=%s", skip_reason)
            return False
        _INT8KV_FA_PREFILL_USED += 1
        if _INT8KV_FA_PREFILL_USED <= 4 or _INT8KV_FA_PREFILL_USED % 64 == 0:
            logger.info(
                "INT8 KV FlashInfer prefill used count=%d backend=%s tokens=%d kv_tokens=%d bridge=%s max_q_len=%d heads=%d kv_heads=%d head_dim=%d",
                _INT8KV_FA_PREFILL_USED,
                _INT8KV_FI_PREFILL_BACKEND,
                num_actual_tokens,
                seq_len if use_continuation_bridge else num_actual_tokens,
                "dequant" if use_continuation_bridge else "raw",
                attn_metadata.max_query_len,
                query.shape[1],
                key.shape[1],
                self.head_size,
        )
        return True

    def _try_gemma4_sm75_sdpa_prefill512(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        num_actual_tokens: int,
    ) -> bool:
        global _GEMMA4_SM75_SDPA_PREFILL512_USED

        if not _GEMMA4_SM75_SDPA_PREFILL512:
            return False
        if (
            output_block_scale is not None
            or output_scale is not None
            or attn_metadata.use_cascade
            or self.attn_type != AttentionType.DECODER
            or self.head_size < 512
            or self.sliding_window[0] >= 0
            or self.logits_soft_cap != 0
            or self.sinks is not None
            or self._is_per_token_head_quant
            or is_quantized_kv_cache(self.kv_cache_dtype)
            or attn_metadata.max_query_len <= 1
            or attn_metadata.max_query_len != attn_metadata.max_seq_len
            or int(attn_metadata.seq_lens.shape[0]) != 1
            or num_actual_tokens != query.shape[0]
            or key.shape[0] != query.shape[0]
            or value.shape[0] != query.shape[0]
        ):
            return False

        q = query[:num_actual_tokens].transpose(0, 1).unsqueeze(0).contiguous()
        k = key[:num_actual_tokens].transpose(0, 1).unsqueeze(0).contiguous()
        v = value[:num_actual_tokens].transpose(0, 1).unsqueeze(0).contiguous()

        enable_gqa = False
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1] != 0:
                return False
            enable_gqa = _GEMMA4_SM75_SDPA_PREFILL512_GQA
            if not enable_gqa:
                repeat = q.shape[1] // k.shape[1]
                if _GEMMA4_SM75_SDPA_PREFILL512_EXPAND:
                    bsz, kv_heads, seq_len, head_dim = k.shape
                    k = (
                        k[:, :, None]
                        .expand(bsz, kv_heads, repeat, seq_len, head_dim)
                        .reshape(bsz, q.shape[1], seq_len, head_dim)
                    )
                    v = (
                        v[:, :, None]
                        .expand(bsz, kv_heads, repeat, seq_len, head_dim)
                        .reshape(bsz, q.shape[1], seq_len, head_dim)
                    )
                else:
                    k = k.repeat_interleave(repeat, dim=1)
                    v = v.repeat_interleave(repeat, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
            enable_gqa=enable_gqa,
        )
        output[:num_actual_tokens].copy_(attn_output.squeeze(0).transpose(0, 1))

        _GEMMA4_SM75_SDPA_PREFILL512_USED += 1
        if _GEMMA4_SM75_SDPA_PREFILL512_USED <= 4:
            logger.info(
                "Gemma4 SM75 SDPA prefill512 used count=%d tokens=%d heads=%d "
                "kv_heads=%d head_dim=%d gqa=%s",
                _GEMMA4_SM75_SDPA_PREFILL512_USED,
                num_actual_tokens,
                q.shape[1],
                key.shape[1],
                self.head_size,
                enable_gqa,
        )
        return True

    def _try_gemma4_sm75_flashinfer_prefill256(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
        num_actual_tokens: int,
    ) -> bool:
        global _GEMMA4_SM75_FI_PREFILL256_USED

        if not _GEMMA4_SM75_FI_PREFILL256:
            return False
        if (
            output_block_scale is not None
            or output_scale is not None
            or attn_metadata.use_cascade
            or self.attn_type != AttentionType.DECODER
            or self.head_size != 256
            or self.sliding_window[0] <= 0
            or self.logits_soft_cap != 0
            or self.sinks is not None
            or self._is_per_token_head_quant
            or is_quantized_kv_cache(self.kv_cache_dtype)
            or attn_metadata.max_query_len <= 1
            or attn_metadata.max_query_len != attn_metadata.max_seq_len
            or int(attn_metadata.seq_lens.shape[0]) != 1
            or attn_metadata.query_start_loc_cpu is None
            or num_actual_tokens != query.shape[0]
            or key.shape[0] != query.shape[0]
            or value.shape[0] != query.shape[0]
            or query.dtype != torch.float16
            or key.dtype != torch.float16
            or value.dtype != torch.float16
        ):
            return False

        indptr_cpu = attn_metadata.query_start_loc_cpu.to(torch.int32)
        plan_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            str(query.dtype),
            str(key.dtype),
            tuple(int(x) for x in indptr_cpu.tolist()),
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            self.sliding_window[0],
        )
        wrapper = self._get_or_plan_gemma4_flashinfer_prefill_wrapper(
            query.device,
            plan_key,
            {
                "qo_indptr": self._flashinfer_indptr(
                    indptr_cpu,
                    self.num_heads,
                    self.head_size,
                    _GEMMA4_SM75_FI_PREFILL256_BACKEND,
                ),
                "kv_indptr": self._flashinfer_indptr(
                    indptr_cpu,
                    self.num_kv_heads,
                    self.head_size,
                    _GEMMA4_SM75_FI_PREFILL256_BACKEND,
                ),
                "num_qo_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim_qk": self.head_size,
                "causal": True,
                "window_left": self.sliding_window[0],
                "logits_soft_cap": self.logits_soft_cap,
                "sm_scale": self.scale,
                "pos_encoding_mode": "NONE",
                "q_data_type": query.dtype,
                "kv_data_type": key.dtype,
                "seq_lens": indptr_cpu[1:] - indptr_cpu[:-1],
                "seq_lens_q": indptr_cpu[1:] - indptr_cpu[:-1],
                "max_token_per_sequence": attn_metadata.max_query_len,
                "max_sequence_kv": attn_metadata.max_seq_len,
            },
        )

        q_prefill = query[:num_actual_tokens]
        k_prefill = key[:num_actual_tokens]
        v_prefill = value[:num_actual_tokens]
        if not q_prefill.is_contiguous():
            q_prefill = q_prefill.contiguous()
        if not k_prefill.is_contiguous():
            k_prefill = k_prefill.contiguous()
        if not v_prefill.is_contiguous():
            v_prefill = v_prefill.contiguous()

        out = wrapper.run(q_prefill, k_prefill, v_prefill)
        output[:num_actual_tokens].copy_(out)

        _GEMMA4_SM75_FI_PREFILL256_USED += 1
        if _GEMMA4_SM75_FI_PREFILL256_USED <= 4:
            logger.info(
                "Gemma4 SM75 FlashInfer prefill256 used count=%d backend=%s "
                "tokens=%d heads=%d kv_heads=%d head_dim=%d window=%s",
                _GEMMA4_SM75_FI_PREFILL256_USED,
                _GEMMA4_SM75_FI_PREFILL256_BACKEND,
                num_actual_tokens,
                query.shape[1],
                key.shape[1],
                self.head_size,
                self.sliding_window,
            )
        return True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        if self._try_gemma4_sm75_sdpa_prefill512(
            query,
            key,
            value,
            output,
            attn_metadata,
            output_scale,
            output_block_scale,
            num_actual_tokens,
        ):
            return output

        if self._try_gemma4_sm75_flashinfer_prefill256(
            query,
            key,
            value,
            output,
            attn_metadata,
            output_scale,
            output_block_scale,
            num_actual_tokens,
        ):
            return output

        if self._try_int8kv_fa_prefill(
            query,
            key,
            value,
            kv_cache,
            output,
            attn_metadata,
            num_actual_tokens,
            output_scale,
            output_block_scale,
        ):
            return output

        # Per-token-head quantized KV cache: use separate scale caches.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
        # FP8 per-tensor / INT8 per-tensor / auto path (original flow).
        else:
            key_cache, value_cache = kv_cache.unbind(1)
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                # [FORK-PORT] PR#41505: int8_per_tensor is already int8,
                # no fp8 view needed
                and self.kv_cache_dtype != "int8_per_tensor"
            ):
                if key_cache.dtype != self.fp8_dtype:
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)
                assert layer._q_scale_float == 1.0, (
                    "A non 1.0 q_scale is not currently supported."
                )
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
        )

        if self._is_per_token_head_quant and num_actual_tokens == 1:
            self._debug_compare_single_token_decode(
                layer,
                query[:num_actual_tokens],
                output[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata,
            )

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            # The per-token-head cache writer only carries explicit strides for
            # token/head axes; keep the head_dim axis materialized as a
            # canonical contiguous layout before quantizing new KV entries.
            if not key.is_contiguous():
                key = key.contiguous()
            if not value.is_contiguous():
                value = value.contiguous()
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                slot_mapping,
            )
            self._debug_verify_per_token_head_cache_update(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        key_cache, value_cache = kv_cache.unbind(1)
        if (
            is_quantized_kv_cache(self.kv_cache_dtype)
            # [FORK-PORT] PR#41505: int8_per_tensor is already int8
            and self.kv_cache_dtype != "int8_per_tensor"
        ):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        if self._is_per_token_head_quant:
            if not key.is_contiguous():
                key = key.contiguous()
            if not value.is_contiguous():
                value = value.contiguous()

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
