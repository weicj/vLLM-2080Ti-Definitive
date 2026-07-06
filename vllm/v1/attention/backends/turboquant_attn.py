# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _fp8_format_code,
    _fp8_format_name,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = None  # type: ignore[assignment]

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
except ImportError:
    BatchDecodeWithPagedKVCacheWrapper = None  # type: ignore[assignment]


def _flashinfer_version() -> Version | None:
    try:
        return Version(version("flashinfer-python"))
    except (PackageNotFoundError, ValueError):
        return None


_FLASHINFER_VERSION = _flashinfer_version()

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128
_SPEC_CONTINUATION_DECODE_FASTPATH = (
    os.getenv("VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH", "0") == "1"
)


def _normalize_tq_prefix_combine_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on", "force", "always"):
        return "on"
    if normalized in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    if normalized in ("", "auto"):
        return "auto"
    raise ValueError(
        "VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE must be one of "
        "off, on, auto, 0, or 1"
    )


_TQ_CONTINUATION_PREFIX_COMBINE_MODE = _normalize_tq_prefix_combine_mode(
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE", "off")
)
_TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS = max(
    0,
    int(os.getenv("VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS", "20480")),
)


def _tq_continuation_prefix_combine_enabled(seq_len: int) -> bool:
    if _TQ_CONTINUATION_PREFIX_COMBINE_MODE == "on":
        return True
    if _TQ_CONTINUATION_PREFIX_COMBINE_MODE == "auto":
        return seq_len >= _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS
    return False


def _normalize_turboquant_flashinfer_backend(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return "fa2"
    return normalized or "fa2"


_DEFAULT_TQ_FI_BACKEND = _normalize_turboquant_flashinfer_backend(
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_BACKEND", "fa2")
)
_DEFAULT_TQ_FI_PREFILL = os.getenv("VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL", "1") == "1"
_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK = (
    os.getenv("VLLM_GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK", "1") == "1"
)
_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK = (
    os.getenv("VLLM_GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK", "0") == "1"
)
_TQ_FORCE_DECODE_SDPA = (
    os.getenv("VLLM_TURBOQUANT_FORCE_DECODE_SDPA", "0") == "1"
)
_TQ_FORCE_CONTINUATION_SDPA = (
    os.getenv("VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA", "0") == "1"
)
_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK = (
    os.getenv("VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK", "0") == "1"
)
_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE = (
    os.getenv("VLLM_GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE", "0") == "1"
)
_SM75_TQ_FI_PREFILL_SUPPORTED = (
    _FLASHINFER_VERSION is not None
    and _FLASHINFER_VERSION >= Version("0.6.8")
    and _FLASHINFER_VERSION < Version("0.6.9")
)
_SM75_TQ_FI_PREFILL_MIN_HEAD_DIM_DEFAULT = (
    "0" if _SM75_TQ_FI_PREFILL_SUPPORTED else "1024"
)
_SM75_TQ_FI_PREFILL_MIN_QUERY_LEN_DEFAULT = (
    "1" if _SM75_TQ_FI_PREFILL_SUPPORTED else "128"
)
_SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN_DEFAULT = (
    "1" if _SM75_TQ_FI_PREFILL_SUPPORTED else "4096"
)
_SM75_TQ_FI_PREFILL_MIN_HEAD_DIM = int(
    os.getenv(
        "VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM",
        _SM75_TQ_FI_PREFILL_MIN_HEAD_DIM_DEFAULT,
    )
)
_SM75_TQ_FI_PREFILL_MIN_QUERY_LEN = int(
    os.getenv(
        "VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_QUERY_LEN",
        _SM75_TQ_FI_PREFILL_MIN_QUERY_LEN_DEFAULT,
    )
)
_SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN = int(
    os.getenv(
        "VLLM_TURBOQUANT_SM75_FLASHINFER_CONTINUATION_MIN_QUERY_LEN",
        _SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN_DEFAULT,
    )
)
_DEFAULT_TQ_FI_PLAN_CACHE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE", "1") == "1"
)
_TQ_FI_PREFILL_CUDAGRAPH_SAFE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE", "0") == "1"
)
_TQ_CONTINUATION_WORKSPACE_RESERVE_TOKENS = int(
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS", "0")
)
_TQ_CONTINUATION_SDPA_Q_CHUNK = int(
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK", "0")
)
_TQ_CONTINUATION_SDPA_MAX_QK_CELLS = int(
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS", "0")
)
_TQ_FORCE_DECODE_SDPA_MAX_QK_CELLS = int(
    os.getenv("VLLM_TURBOQUANT_FORCE_DECODE_SDPA_MAX_QK_CELLS", "131072")
)
_TQ_CUDAGRAPH_SPEC_DECODE_SAFE = (
    os.getenv("VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE", "0") == "1"
)
_TQ_DEBUG_MIXED = os.getenv("VLLM_TURBOQUANT_DEBUG_MIXED", "0") == "1"
_SKIP_PREFILL_STORE_FOR_PROFILING = (
    os.getenv("VLLM_TURBOQUANT_SKIP_PREFILL_STORE", "0") == "1"
)


def _read_tq_max_kv_splits_override() -> int | None:
    value = os.getenv("VLLM_TURBOQUANT_MAX_KV_SPLITS")
    if value is None or value == "":
        return None
    try:
        splits = int(value)
    except ValueError as err:
        raise ValueError(
            "VLLM_TURBOQUANT_MAX_KV_SPLITS must be a positive integer"
        ) from err
    if splits <= 0:
        raise ValueError("VLLM_TURBOQUANT_MAX_KV_SPLITS must be a positive integer")
    return splits


_TQ_MAX_KV_SPLITS_OVERRIDE = _read_tq_max_kv_splits_override()
_GEMMA4_TQ4NC_DEBUG_CONTINUATION = (
    os.getenv("VLLM_GEMMA4_TQ4NC_DEBUG_CONTINUATION", "0") == "1"
)
_GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT = (
    os.getenv("VLLM_GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT", "0") == "1"
)
_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER = (
    _GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT
    and os.getenv("VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER", "0") == "1"
)
_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER_FORWARD_PLAN = (
    _GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER
    and os.getenv(
        "VLLM_GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER_FORWARD_PLAN", "0"
    )
    == "1"
)
_GEMMA4_TQ4NC_SHARED_FP16_TRITON = (
    _GEMMA4_TQ4NC_SHARED_FP16_UNSAFE_EXPERIMENT
    and os.getenv("VLLM_GEMMA4_TQ4NC_SHARED_FP16_TRITON", "0") == "1"
)
_TQ_FI_PREFILL_WORKSPACES: dict[tuple[str, str], torch.Tensor] = {}
_TQ_FI_PREFILL_WRAPPERS: dict[tuple[Any, ...], Any] = {}
logger = init_logger(__name__)


@triton.jit
def _shared_fp16_paged_decode_kernel(
    Q,
    KV,
    BT,
    OUT,
    stride_kv_block: tl.constexpr,
    stride_kv_kv: tl.constexpr,
    stride_kv_page: tl.constexpr,
    stride_kv_h: tl.constexpr,
    stride_bt_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    SCALE: tl.constexpr,
):
    hq = tl.program_id(0)
    kv_group: tl.constexpr = Hq // Hkv
    hkv = hq // kv_group

    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    q = tl.load(Q + hq * D + d, mask=d_mask, other=0.0).to(tl.float32)

    start: tl.constexpr = 0
    if WINDOW_LEFT >= 0 and SEQ_LEN > WINDOW_LEFT + 1:
        start = SEQ_LEN - WINDOW_LEFT - 1

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for base in range(start, SEQ_LEN, BLOCK_M):
        pos = base + tl.arange(0, BLOCK_M)
        p_mask = pos < SEQ_LEN
        page_idx = pos // PAGE_SIZE
        page_off = pos - page_idx * PAGE_SIZE
        block_num = tl.load(BT + page_idx, mask=p_mask, other=0).to(tl.int64)
        kv_base = (
            block_num[:, None] * stride_kv_block
            + page_off[:, None] * stride_kv_page
            + hkv * stride_kv_h
            + d[None, :]
        )
        k = tl.load(KV + kv_base, mask=p_mask[:, None] & d_mask[None, :], other=0.0)
        scores = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * SCALE
        scores = tl.where(p_mask, scores, -float("inf"))

        new_m = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m)

        v_base = (
            block_num[:, None] * stride_kv_block
            + stride_kv_kv
            + page_off[:, None] * stride_kv_page
            + hkv * stride_kv_h
            + d[None, :]
        )
        v = tl.load(KV + v_base, mask=p_mask[:, None] & d_mask[None, :], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = new_m

    out = acc / tl.maximum(l, 1.0e-20)
    tl.store(OUT + hq * stride_out_h + d, out, mask=d_mask)


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


def _normalize_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_shared_flashinfer_prefill_workspace(
    device: torch.device,
    backend: str,
) -> torch.Tensor:
    device = _normalize_cuda_device(device)
    key = (str(device), backend)
    workspace = _TQ_FI_PREFILL_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.empty(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _TQ_FI_PREFILL_WORKSPACES[key] = workspace
    return workspace


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    seq_lens_cpu: torch.Tensor  # (num_reqs,) — CPU copy for graph-safe host ops
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    query_start_loc_cpu: torch.Tensor  # (num_reqs + 1,) — CPU copy for host ops
    query_start_loc_cpu_pinned: torch.Tensor | None = None
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    force_spec_decode: bool = False


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=_TQ_CUDAGRAPH_SPEC_DECODE_SAFE
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        if (
            _TQ_CUDAGRAPH_SPEC_DECODE_SAFE
            and 1 < attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
        ):
            attn_metadata.force_spec_decode = True
            attn_metadata.seq_lens.fill_(attn_metadata.max_query_len)
            attn_metadata.seq_lens_cpu.fill_(attn_metadata.max_query_len)
            return attn_metadata
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            seq_lens_cpu=cam.seq_lens_cpu,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            query_start_loc_cpu_pinned=cam.query_start_loc_cpu.pin_memory(),
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        # Decode over the actual local-attention window for sliding layers.
        # Gemma4 has many sliding layers; scanning full context here is a
        # major decode regression versus the non-TQ attention backends.
        self._decode_sliding_window = 0 if sliding_window is None else int(sliding_window)
        self._prefill_sliding_window = (
            -1 if sliding_window is None else max(0, int(sliding_window) - 1)
        )

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1
        if cfg.key_fp8:
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            fp8_override = os.getenv(
                "VLLM_TURBOQUANT_K8V4_FP8_FORMAT",
                "auto",
            ).strip().lower() or "auto"
            logger.info_once(
                "TurboQuant raw FP8 key format is %s "
                "(VLLM_TURBOQUANT_K8V4_FP8_FORMAT=%s)",
                _fp8_format_name(device_index),
                fp8_override,
            )

        self._fi_prefill_workspace = None
        self._fi_prefill_backend = _DEFAULT_TQ_FI_BACKEND
        self._fi_single_qo_indptr_cpu = None
        self._fi_single_kv_indptr_cpu = None
        self._decode_workspace_reserved = False
        self._continuation_workspace_reserved = False
        self._shared_draft_sdpa_notice_logged = False
        self._shared_fp16_decode_workspace = None
        self._shared_fp16_decode_wrappers: dict[tuple[Any, ...], Any] = {}
        capability = current_platform.get_device_capability()
        self._sm75_tq_prefill_guard = (
            capability is not None
            and capability.major == 7
            and capability.minor == 5
        )
        sm75_skip_flashinfer_prefill = (
            self._sm75_tq_prefill_guard
            and head_size < _SM75_TQ_FI_PREFILL_MIN_HEAD_DIM
        )
        self._sm75_skip_flashinfer_prefill = sm75_skip_flashinfer_prefill
        self._use_flashinfer_prefill = (
            _DEFAULT_TQ_FI_PREFILL
            and
            BatchPrefillWithRaggedKVCacheWrapper is not None
            and current_platform.is_cuda()
            and not sm75_skip_flashinfer_prefill
        )
        # Detect flash-attn version (FA2/3/4) for prefill paths.
        self.fa_version = (
            None
            if (
                self._use_flashinfer_prefill
                or sm75_skip_flashinfer_prefill
            )
            else get_flash_attn_version(head_size=head_size)
        )
        if self._use_flashinfer_prefill:
            cap_str = capability.as_version_str() if capability is not None else "unknown"
            logger.info_once(
                "TurboQuant prefill is using FlashInfer backend=%s on CUDA capability %s (flashinfer=%s)",
                self._fi_prefill_backend,
                cap_str,
                _FLASHINFER_VERSION or "unknown",
            )
        elif sm75_skip_flashinfer_prefill:
            logger.info_once(
                "TurboQuant prefill is using SDPA fallback on CUDA capability 7.5 for head_dim=%s because FlashInfer prefill is disabled for this head_dim or FlashInfer version",
                head_size,
            )

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            _TQ_MAX_KV_SPLITS_OVERRIDE
            or vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        if _TQ_MAX_KV_SPLITS_OVERRIDE is not None:
            logger.info_once(
                "TurboQuant decode max KV splits forced to %s by "
                "VLLM_TURBOQUANT_MAX_KV_SPLITS",
                self.max_num_kv_splits,
            )
        if _TQ_FORCE_DECODE_SDPA:
            logger.info_once(
                "TurboQuant decode is forced to full-dequant SDPA fallback by "
                "VLLM_TURBOQUANT_FORCE_DECODE_SDPA=1"
            )
        if _TQ_FORCE_CONTINUATION_SDPA:
            logger.info_once(
                "TurboQuant continuation prefill is forced to full-dequant "
                "SDPA fallback by VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA=1"
            )
        if _TQ_CONTINUATION_PREFIX_COMBINE_MODE != "off":
            logger.info_once(
                "TurboQuant continuation prefix-combine experiment is enabled "
                "with mode=%s min_tokens=%s",
                _TQ_CONTINUATION_PREFIX_COMBINE_MODE,
                _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS,
            )

    def _get_flashinfer_prefill_wrapper(self, device: torch.device):
        if not self._use_flashinfer_prefill:
            return None

        if self._fi_prefill_workspace is None:
            if _DEFAULT_TQ_FI_PLAN_CACHE:
                self._fi_prefill_workspace = _get_shared_flashinfer_prefill_workspace(
                    device, self._fi_prefill_backend
                )
            else:
                workspace_bytes = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
                self._fi_prefill_workspace = torch.empty(
                    workspace_bytes,
                    dtype=torch.uint8,
                    device=device,
                )

        return BatchPrefillWithRaggedKVCacheWrapper(
            self._fi_prefill_workspace,
            "NHD",
            backend=self._fi_prefill_backend,
        )

    def _get_or_plan_flashinfer_prefill_wrapper(
        self,
        device: torch.device,
        plan_key: tuple[Any, ...],
        plan_kwargs: dict[str, Any],
    ):
        if not self._use_flashinfer_prefill:
            return None
        if not _DEFAULT_TQ_FI_PLAN_CACHE:
            wrapper = self._get_flashinfer_prefill_wrapper(device)
            assert wrapper is not None
            wrapper.plan(**plan_kwargs)
            return wrapper

        norm_device = _normalize_cuda_device(device)
        cache_key = (str(norm_device), self._fi_prefill_backend, *plan_key)
        wrapper = _TQ_FI_PREFILL_WRAPPERS.get(cache_key)
        if wrapper is None:
            workspace = _get_shared_flashinfer_prefill_workspace(
                norm_device, self._fi_prefill_backend
            )
            wrapper_kwargs: dict[str, Any] = {"backend": self._fi_prefill_backend}
            if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
                qo_indptr = plan_kwargs.get("qo_indptr")
                kv_indptr = plan_kwargs.get("kv_indptr")
                if qo_indptr is None:
                    raise RuntimeError("FlashInfer cudagraph-safe prefill requires qo_indptr")
                if kv_indptr is None:
                    raise RuntimeError("FlashInfer cudagraph-safe prefill requires kv_indptr")
                qo_buf = torch.empty(
                    tuple(qo_indptr.shape), dtype=qo_indptr.dtype, device=norm_device
                )
                kv_buf = torch.empty(
                    tuple(kv_indptr.shape), dtype=kv_indptr.dtype, device=norm_device
                )
                wrapper_kwargs.update(
                    {
                        "use_cuda_graph": True,
                        "qo_indptr_buf": qo_buf,
                        "kv_indptr_buf": kv_buf,
                    }
                )
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                "NHD",
                **wrapper_kwargs,
            )
            wrapper.plan(**plan_kwargs)
            _TQ_FI_PREFILL_WRAPPERS[cache_key] = wrapper
        return wrapper

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self._use_flashinfer_prefill and self._fi_prefill_workspace is None:
            device = torch.device("cuda", torch.cuda.current_device())
            if _DEFAULT_TQ_FI_PLAN_CACHE:
                self._fi_prefill_workspace = _get_shared_flashinfer_prefill_workspace(
                    device, self._fi_prefill_backend
                )
            else:
                workspace_bytes = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
                self._fi_prefill_workspace = torch.empty(
                    workspace_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
        self._reserve_decode_workspace(act_dtype)
        self._reserve_continuation_workspace()

    def _reserve_decode_workspace(self, out_dtype: torch.dtype) -> None:
        if self._decode_workspace_reserved or not is_workspace_manager_initialized():
            return
        Hq = self.num_heads
        D = self.head_size
        S = self.max_num_kv_splits
        # Reserve for maximum decode batch size. With MTP + concurrent decode,
        # runtime decode workspace can require B > 1.
        vllm_config = get_current_vllm_config()
        scheduler_cfg = getattr(vllm_config, "scheduler_config", None)
        max_seqs = (
            getattr(scheduler_cfg, "max_num_seqs", 1)
            if scheduler_cfg is not None
            else 1
        )
        B = max(1, int(max_seqs))
        current_workspace_manager().reserve_simultaneous_for_all_ubatches(
            ((B, Hq, S, D + 1), torch.float32),
            ((B, Hq, D), out_dtype),
            ((B, Hq), torch.float32),
        )
        self._decode_workspace_reserved = True

    def _reserve_continuation_workspace(self) -> None:
        if (
            self._continuation_workspace_reserved
            or not is_workspace_manager_initialized()
        ):
            return

        vllm_config = get_current_vllm_config()
        scheduler_cfg = getattr(vllm_config, "scheduler_config", None)
        cache_cfg = getattr(vllm_config, "cache_config", None)

        max_batched_tokens = int(
            getattr(scheduler_cfg, "max_num_batched_tokens", 0)
            if scheduler_cfg is not None
            else 0
        )

        # Continuation prefill enters the large dequant path only when q_len
        # exceeds this threshold.
        if max_batched_tokens <= _CONTINUATION_DECODE_THRESHOLD:
            self._continuation_workspace_reserved = True
            return

        block_size = int(
            getattr(cache_cfg, "block_size", 0) if cache_cfg is not None else 0
        )
        if block_size <= 0:
            block_size = max_batched_tokens

        reserve_tokens = max(
            max_batched_tokens, _TQ_CONTINUATION_WORKSPACE_RESERVE_TOKENS
        )
        reserve_cached_len = math.ceil(reserve_tokens / block_size) * block_size
        if reserve_cached_len <= 0:
            self._continuation_workspace_reserved = True
            return

        buf_shape = (
            1,
            self.num_kv_heads,
            reserve_cached_len,
            self.head_size,
        )
        current_workspace_manager().reserve_simultaneous_for_all_ubatches(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        self._continuation_workspace_reserved = True

    def _has_flash_attn_prefill(self) -> bool:
        if self._sm75_skip_flashinfer_prefill:
            return False
        return _HAS_FLASH_ATTN

    def _flashinfer_indptr(
        self,
        indptr: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        if self._fi_prefill_backend == "cudnn":
            return indptr * (num_heads * head_dim)
        return indptr

    def _use_flashinfer_for_first_chunk(self, query_len: int) -> bool:
        if not self._use_flashinfer_prefill:
            return False
        if not self._sm75_tq_prefill_guard:
            return True
        return query_len >= _SM75_TQ_FI_PREFILL_MIN_QUERY_LEN

    def _use_flashinfer_for_continuation(self, query_len: int) -> bool:
        if not self._use_flashinfer_prefill:
            return False
        if not self._sm75_tq_prefill_guard:
            return True
        return query_len >= _SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN

    def _shared_fp16_decode_flashinfer(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        layer: torch.nn.Module,
    ) -> torch.Tensor | None:
        if (
            not _GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER_FORWARD_PLAN
            or BatchDecodeWithPagedKVCacheWrapper is None
            or cached_len <= 0
        ):
            return None
        q_len, Hq, D = query.shape
        if q_len != 1 or kv_cache.dim() != 5:
            return None
        device = query.device
        block_size = kv_cache.shape[2]
        cache_hk = kv_cache.shape[3]
        if Hq % cache_hk != 0:
            return None

        if self._shared_fp16_decode_workspace is None:
            workspace_bytes = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            self._shared_fp16_decode_workspace = torch.empty(
                workspace_bytes,
                dtype=torch.uint8,
                device=device,
            )

        pages = math.ceil(cached_len / block_size)
        indptr_cpu = torch.tensor([0, pages], dtype=torch.int32, pin_memory=True)
        indices = block_table[0, :pages].contiguous()
        last_page_len = cached_len % block_size
        if last_page_len == 0:
            last_page_len = block_size
        last_page_len_cpu = torch.tensor(
            [last_page_len], dtype=torch.int32, pin_memory=True
        )

        plan_key = (
            str(device),
            Hq,
            cache_hk,
            D,
            block_size,
            self._prefill_sliding_window,
            str(query.dtype),
            str(kv_cache.dtype),
        )
        wrapper = self._shared_fp16_decode_wrappers.get(plan_key)
        if wrapper is None:
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._shared_fp16_decode_workspace,
                "NHD",
                use_tensor_cores=False,
                backend="auto",
            )
            self._shared_fp16_decode_wrappers[plan_key] = wrapper
        wrapper.plan(
            indptr_cpu,
            indices,
            last_page_len_cpu,
            num_qo_heads=Hq,
            num_kv_heads=cache_hk,
            head_dim=D,
            page_size=block_size,
            pos_encoding_mode="NONE",
            window_left=self._prefill_sliding_window,
            q_data_type=query.dtype,
            kv_data_type=kv_cache.dtype,
            o_data_type=query.dtype,
            sm_scale=self.scale,
        )

        out = wrapper.run(query, kv_cache)
        if _GEMMA4_TQ4NC_DEBUG_CONTINUATION:
            logger.warning(
                "Gemma4 shared FP16 FlashInfer decode: layer=%s "
                "shared_target=%s cached_len=%s Hq=%s cache_hk=%s D=%s "
                "block_size=%s pages=%s",
                getattr(layer, "layer_name", None),
                getattr(layer, "kv_sharing_target_layer_name", None),
                cached_len,
                Hq,
                cache_hk,
                D,
                block_size,
                pages,
            )
        return out

    def _shared_fp16_decode_triton(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        layer: torch.nn.Module,
    ) -> torch.Tensor | None:
        if (
            not _GEMMA4_TQ4NC_SHARED_FP16_TRITON
            or seq_len <= 0
            or query.shape[0] != 1
            or kv_cache.dim() != 5
        ):
            return None
        _, Hq, D = query.shape
        Hkv = kv_cache.shape[3]
        if Hkv <= 0 or Hq % Hkv != 0:
            return None
        page_size = kv_cache.shape[2]
        pages = math.ceil(seq_len / page_size)
        if pages > block_table.shape[1]:
            return None

        out = torch.empty_like(query)
        block_m = 8 if D >= 512 else 16
        window_left = self._decode_sliding_window
        _shared_fp16_paged_decode_kernel[(Hq,)](
            query,
            kv_cache,
            block_table[0],
            out,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            BLOCK_D=triton.next_power_of_2(D),
            BLOCK_M=block_m,
            PAGE_SIZE=page_size,
            SEQ_LEN=seq_len,
            WINDOW_LEFT=window_left,
            SCALE=self.scale,
            num_warps=8 if D >= 512 else 4,
        )
        if _GEMMA4_TQ4NC_DEBUG_CONTINUATION:
            logger.warning(
                "Gemma4 shared FP16 Triton decode: layer=%s shared_target=%s "
                "seq_len=%s Hq=%s Hkv=%s D=%s page_size=%s pages=%s",
                getattr(layer, "layer_name", None),
                getattr(layer, "kv_sharing_target_layer_name", None),
                seq_len,
                Hq,
                Hkv,
                D,
                page_size,
                pages,
            )
        return out

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        # fa_utils.get_flash_attn_version() returns None on backends that
        # should not pass an explicit fa_version kwarg.
        if self.fa_version is None:
            return flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            fa_version=self.fa_version,
        )

    def _ensure_on_device(self, layer, device):
        """One-time derivation of TQ buffers (rotation matrix, midpoints).

        The Hadamard rotation is shared across all layers: random sign
        flips do not improve Lloyd-Max quantization quality because the
        quantizer is symmetric around zero (sign-flipping a coordinate
        maps it to the mirror centroid with identical distortion).
        """
        if not hasattr(layer, "_tq_cached"):
            D = self.head_size

            # Pure Hadamard: orthonormal + symmetric (H = H^T), enabling
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            # fp16 copy for rotation in continuation prefill path
            layer._tq_Pi_half = H.to(torch.float16)

            # Centroids for Lloyd-Max quantization.
            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
                device=device, dtype=torch.float32
            )

            c_sorted, _ = layer._tq_centroids.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        if _SKIP_PREFILL_STORE_FOR_PROFILING and N > 1:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration).
        # Use Any-typed alias for dynamic _tq_* attrs set by _ensure_on_device.
        tq_layer: Any = layer
        device = q.device
        self._ensure_on_device(tq_layer, device)
        Pi = tq_layer._tq_Pi
        PiT = tq_layer._tq_PiT
        centroids = tq_layer._tq_centroids

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        use_decode_sdpa = self._use_decode_sdpa_fallback()
        use_shared_draft_decode_sdpa = self._use_shared_draft_decode_sdpa_fallback(
            layer
        )
        if attn_metadata.force_spec_decode:
            if _TQ_FORCE_DECODE_SDPA:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._spec_decode_attention_sdpa_fallback(
                    q, k, v, kv_cache, attn_metadata, Pi, centroids, layer
                )
            else:
                attn_out = self._spec_decode_attention(
                    q,
                    kv_cache,
                    attn_metadata,
                    Pi,
                    centroids,
                    PiT,
                    layer,
                )
            if output.ndim == 3:
                output[:N] = attn_out.to(output.dtype)
            else:
                output[:N] = attn_out.reshape(N, -1).to(output.dtype)
            return output
        if (
            use_shared_draft_decode_sdpa
            and not use_decode_sdpa
            and not self._shared_draft_sdpa_notice_logged
        ):
            shared_target = getattr(layer, "kv_sharing_target_layer_name", None)
            logger.info(
                "TurboQuant decode is using shared-draft SDPA fallback "
                "for head_dim=%s target=%s",
                self.head_size,
                shared_target,
            )
            self._shared_draft_sdpa_notice_logged = True

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            if use_decode_sdpa or use_shared_draft_decode_sdpa:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._decode_attention_sdpa_fallback(
                    q, k, v, kv_cache, attn_metadata, Pi, centroids, layer
                )
            else:
                attn_out = self._decode_attention(
                    q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
                )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )
        elif num_decode_tokens >= N or num_decodes >= attn_metadata.seq_lens.shape[0]:
            # Spec decode capture can mark a uniform continuation batch as
            # prefill while split_decodes_and_prefills classifies every request
            # as decode. There is no prefill tail in that case.
            if _TQ_CUDAGRAPH_SPEC_DECODE_SAFE and attn_metadata.max_query_len > 1:
                attn_out = self._spec_decode_attention(
                    q,
                    kv_cache,
                    attn_metadata,
                    Pi,
                    centroids,
                    PiT,
                    layer,
                )
            elif use_decode_sdpa or use_shared_draft_decode_sdpa:
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                attn_out = self._decode_attention_sdpa_fallback(
                    q, k, v, kv_cache, attn_metadata, Pi, centroids, layer
                )
            else:
                attn_out = self._decode_attention(
                    q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
                )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                seq_lens_cpu=attn_metadata.seq_lens_cpu[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                query_start_loc_cpu=attn_metadata.query_start_loc_cpu[
                    : num_decodes + 1
                ],
                query_start_loc_cpu_pinned=(
                    attn_metadata.query_start_loc_cpu_pinned[: num_decodes + 1]
                    if attn_metadata.query_start_loc_cpu_pinned is not None
                    else None
                ),
                num_actual_tokens=num_decode_tokens,
                max_query_len=max(
                    attn_metadata.query_start_loc_cpu[i + 1].item()
                    - attn_metadata.query_start_loc_cpu[i].item()
                    for i in range(num_decodes)
                ),
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            if decode_meta.max_query_len > 1:
                if _TQ_DEBUG_MIXED:
                    logger.warning(
                        "TurboQuant mixed decode spec-path: num_decode_tokens=%s "
                        "num_decodes=%s decode_max_query=%s",
                        num_decode_tokens,
                        num_decodes,
                        decode_meta.max_query_len,
                    )
                if _TQ_FORCE_DECODE_SDPA:
                    k_dec = key[:num_decode_tokens].view(
                        num_decode_tokens, self.num_kv_heads, self.head_size
                    )
                    v_dec = value[:num_decode_tokens].view(
                        num_decode_tokens, self.num_kv_heads, self.head_size
                    )
                    attn_out[:num_decode_tokens] = (
                        self._spec_decode_attention_sdpa_fallback(
                            q[:num_decode_tokens],
                            k_dec,
                            v_dec,
                            kv_cache,
                            decode_meta,
                            Pi,
                            centroids,
                            layer,
                        )
                    )
                else:
                    attn_out[:num_decode_tokens] = self._spec_decode_attention(
                        q[:num_decode_tokens],
                        kv_cache,
                        decode_meta,
                        Pi,
                        centroids,
                        PiT,
                        layer,
                    )
            elif use_decode_sdpa or use_shared_draft_decode_sdpa:
                k_dec = key[:num_decode_tokens].view(
                    num_decode_tokens, self.num_kv_heads, self.head_size
                )
                v_dec = value[:num_decode_tokens].view(
                    num_decode_tokens, self.num_kv_heads, self.head_size
                )
                attn_out[:num_decode_tokens] = self._decode_attention_sdpa_fallback(
                    q[:num_decode_tokens],
                    k_dec,
                    v_dec,
                    kv_cache,
                    decode_meta,
                    Pi,
                    centroids,
                    layer,
                )
            else:
                attn_out[:num_decode_tokens] = self._decode_attention(
                    q[:num_decode_tokens],
                    kv_cache,
                    decode_meta,
                    Pi,
                    centroids,
                    PiT,
                    layer,
                )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            prefill_query_lens_cpu = (
                attn_metadata.query_start_loc_cpu[num_decodes + 1 :]
                - attn_metadata.query_start_loc_cpu[num_decodes:-1]
            )
            # Use CPU-side max to avoid GPU→CPU sync from .item()
            prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())
            prefill_max_query = max(prefill_query_lens_cpu.tolist())
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            if _TQ_DEBUG_MIXED:
                logger.warning(
                    "TurboQuant mixed batch: N=%s num_decodes=%s "
                    "num_decode_tokens=%s decode_max_query=%s "
                    "prefill_reqs=%s prefill_max_query=%s "
                    "prefill_max_seq=%s full_max_query=%s full_max_seq=%s",
                    N,
                    num_decodes,
                    num_decode_tokens,
                    decode_meta.max_query_len,
                    prefill_seq_lens.shape[0],
                    prefill_max_query,
                    prefill_max_seq,
                    attn_metadata.max_query_len,
                    attn_metadata.max_seq_len,
                )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:],
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                query_start_loc_cpu=(
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                ),
                query_start_loc_cpu_pinned=(
                    attn_metadata.query_start_loc_cpu_pinned[num_decodes:]
                    - num_decode_tokens
                    if attn_metadata.query_start_loc_cpu_pinned is not None
                    else None
                ),
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=prefill_max_query,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
                layer=layer,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _spec_decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        layer: Any,
    ) -> torch.Tensor:
        N, Hq, D = query.shape
        qsl = attn_metadata.query_start_loc_cpu.tolist()
        num_reqs = attn_metadata.seq_lens_cpu.shape[0]
        output = torch.empty_like(query)

        _max_seq = max(attn_metadata.max_seq_len, attn_metadata.max_query_len)
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            rel_seq_lens = _ac[1 : q_len + 1]
            synth_seq_lens = (
                attn_metadata.seq_lens[i : i + 1] - q_len + rel_seq_lens
            ).contiguous()
            synth_bt = (
                attn_metadata.block_table[i : i + 1]
                .expand(q_len, -1)
                .contiguous()
            )
            output[q_start:q_end] = triton_turboquant_decode_attention(
                query=query[q_start:q_end],
                kv_cache=kv_cache,
                block_table=synth_bt,
                seq_lens=synth_seq_lens,
                Pi=Pi,
                centroids=centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                key_fp8=self.tq_config.key_fp8,
                norm_correction=self.tq_config.norm_correction,
                PiT=PiT,
                max_num_kv_splits=self.max_num_kv_splits,
                sliding_window=self._decode_sliding_window,
            ).to(query.dtype)

        return output

    def _spec_decode_attention_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        _, Hq, _ = query.shape
        Hk = key.shape[1]
        qsl = attn_metadata.query_start_loc_cpu.tolist()
        seq_lens = attn_metadata.seq_lens_cpu.tolist()
        output = torch.empty_like(query)

        for i, seq_len in enumerate(seq_lens):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = int(seq_len)
            cached_len = max(seq_len - q_len, 0)
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            if cached_len == 0:
                q_t = q_seq.transpose(0, 1).contiguous()
                k_t = k_seq.transpose(0, 1).contiguous()
                v_t = v_seq.transpose(0, 1).contiguous()
                out = F.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    is_causal=True,
                    scale=self.scale,
                    enable_gqa=(Hk < Hq),
                ).transpose(0, 1)
            else:
                out = self._continuation_prefill(
                    layer,
                    q_seq,
                    k_seq,
                    v_seq,
                    kv_cache,
                    attn_metadata.block_table[i : i + 1],
                    cached_len,
                    seq_len,
                    Pi,
                    centroids,
                    force_sdpa=True,
                )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _spec_continuation_decode_attention(
        self,
        query: torch.Tensor,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cached_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        arange_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fast continuation attention for small MTP chunks.

        The cached prefix can be read through the TQ decode kernel, but the
        current chunk must use the uncompressed K/V produced in this step.
        Reading the current chunk back from the quantized cache corrupts
        multi-token speculative continuation quality.
        """
        if cached_len <= 0 or self._decode_sliding_window > 0:
            return None

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        if Hq % Hk != 0:
            return None

        # Prefix attention from compressed cache. All queries in this chunk see
        # the same cached prefix; causal masking for current tokens is handled
        # below on the raw current K/V.
        prefix_seq_lens = torch.full(
            (q_len,),
            cached_len,
            dtype=arange_cache.dtype,
            device=query.device,
        )
        prefix_bt = block_table.expand(q_len, -1).contiguous()
        prefix_lse = torch.empty(q_len, Hq, dtype=torch.float32, device=query.device)
        prefix_out = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=prefix_bt,
            seq_lens=prefix_seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            output_buf=torch.empty_like(query),
            lse_buf=prefix_lse,
            max_num_kv_splits=self.max_num_kv_splits,
            sliding_window=self._decode_sliding_window,
        )

        # Current chunk attention from raw K/V. This is tiny for MTP
        # continuation (typically q_len <= 8), so torch SDPA-style math is fast
        # enough and avoids quantizing the just-produced K/V before use.
        kv_group_size = Hq // Hk
        q_float = query.float().view(q_len, Hk, kv_group_size, D)
        k_float = key_chunk.float()
        v_float = value_chunk.float()
        scores = torch.einsum("thgd,shd->thgs", q_float, k_float) * self.scale
        idx = torch.arange(q_len, device=query.device)
        causal = idx.view(q_len, 1, 1, 1) >= idx.view(1, 1, 1, q_len)
        scores = scores.masked_fill(~causal, float("-inf"))
        current_lse = torch.logsumexp(scores, dim=-1).reshape(q_len, Hq)
        probs = torch.softmax(scores, dim=-1)
        current_out = torch.einsum("thgs,shd->thgd", probs, v_float)
        current_out = current_out.reshape(q_len, Hq, D)

        combined_lse = torch.logaddexp(prefix_lse, current_lse)
        prefix_weight = torch.exp(prefix_lse - combined_lse).unsqueeze(-1)
        current_weight = torch.exp(current_lse - combined_lse).unsqueeze(-1)
        return (
            prefix_out.float() * prefix_weight
            + current_out.float() * current_weight
        ).to(query.dtype)

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        if (
            self._use_flashinfer_for_first_chunk(attn_metadata.max_query_len)
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
        ):
            query_start_loc = (
                attn_metadata.query_start_loc_cpu_pinned
                if attn_metadata.query_start_loc_cpu_pinned is not None
                else attn_metadata.query_start_loc_cpu
            )
            qo_indptr = self._flashinfer_indptr(query_start_loc, Hq, D)
            kv_indptr = self._flashinfer_indptr(query_start_loc, key.shape[1], D)
            q_seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            plan_key = (
                "batch_first_chunk",
                Hq,
                key.shape[1],
                D,
                self._prefill_sliding_window,
                str(query.dtype),
                str(key.dtype),
                tuple(int(x) for x in query_start_loc.tolist()),
                attn_metadata.max_query_len,
            )
            prefill_wrapper = self._get_or_plan_flashinfer_prefill_wrapper(
                query.device,
                plan_key,
                {
                    "qo_indptr": qo_indptr,
                    "kv_indptr": kv_indptr,
                    "num_qo_heads": Hq,
                    "num_kv_heads": key.shape[1],
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": self._prefill_sliding_window,
                    "sm_scale": self.scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": query.dtype,
                    "kv_data_type": key.dtype,
                    "seq_lens": q_seq_lens,
                    "seq_lens_q": q_seq_lens,
                    "max_token_per_sequence": attn_metadata.max_query_len,
                    "max_sequence_kv": attn_metadata.max_query_len,
                },
            )
            return prefill_wrapper.run(query, key, value)

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if self._has_flash_attn_prefill() and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        query_start_loc_cpu = attn_metadata.query_start_loc_cpu
        num_reqs = query_start_loc_cpu.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        qsl = query_start_loc_cpu.tolist()
        seq_lens_list = attn_metadata.seq_lens_cpu.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls.
        # During CUDA graph capture, host->device copies must come from pinned
        # CPU memory, so update a stable pinned source and copy into the CUDA
        # tensor instead of assigning Python ints directly to device tensors.
        if not hasattr(self, "_cu_2"):
            self._cu_2_cpu = torch.zeros(2, dtype=torch.int32, pin_memory=True)
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        # Cache arange on self (avoid per-call kernel launch).
        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = _ac
        _arange_cache: torch.Tensor = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if self._use_flashinfer_for_first_chunk(q_len):
                    if self._fi_single_qo_indptr_cpu is None:
                        self._fi_single_qo_indptr_cpu = torch.empty(
                            2, dtype=torch.int32, pin_memory=True
                        )
                        self._fi_single_kv_indptr_cpu = torch.empty(
                            2, dtype=torch.int32, pin_memory=True
                        )
                    self._fi_single_qo_indptr_cpu[0] = 0
                    self._fi_single_qo_indptr_cpu[1] = q_len
                    self._fi_single_kv_indptr_cpu[0] = 0
                    self._fi_single_kv_indptr_cpu[1] = q_len
                    qo_indptr_tokens = self._fi_single_qo_indptr_cpu
                    kv_indptr_tokens = qo_indptr_tokens
                    seq_lens = q_len * torch.ones(
                        1, device=query.device, dtype=torch.int32
                    )
                    plan_key = (
                        "single_first_chunk",
                        Hq,
                        Hk,
                        D,
                        self._prefill_sliding_window,
                        str(q_seq.dtype),
                        str(k_seq.dtype),
                        q_len,
                    )
                    prefill_wrapper = self._get_or_plan_flashinfer_prefill_wrapper(
                        query.device,
                        plan_key,
                        {
                            "qo_indptr": self._flashinfer_indptr(
                                qo_indptr_tokens, Hq, D
                            ),
                            "kv_indptr": self._flashinfer_indptr(
                                kv_indptr_tokens, Hk, D
                            ),
                            "num_qo_heads": Hq,
                            "num_kv_heads": Hk,
                            "head_dim_qk": D,
                            "causal": True,
                            "window_left": self._prefill_sliding_window,
                            "sm_scale": self.scale,
                            "pos_encoding_mode": "NONE",
                            "q_data_type": q_seq.dtype,
                            "kv_data_type": k_seq.dtype,
                            "seq_lens": seq_lens,
                            "seq_lens_q": seq_lens,
                            "max_token_per_sequence": q_len,
                            "max_sequence_kv": q_len,
                        },
                    )
                    out = prefill_wrapper.run(q_seq, k_seq, v_seq)
                elif self._has_flash_attn_prefill():
                    self._cu_2_cpu[0] = 0
                    self._cu_2_cpu[1] = q_len
                    self._cu_2.copy_(self._cu_2_cpu, non_blocking=True)
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if _TQ_DEBUG_MIXED:
                    logger.warning(
                        "TurboQuant continuation chunk: req=%s q_len=%s "
                        "seq_len=%s cached_len=%s kv_dim=%s",
                        i,
                        q_len,
                        seq_len,
                        cached_len,
                        kv_cache.dim(),
                    )
                if (
                    q_len <= _CONTINUATION_DECODE_THRESHOLD
                    and kv_cache.dim() == 5
                    and (
                        _GEMMA4_TQ4NC_SHARED_FP16_TRITON
                        or _GEMMA4_TQ4NC_SHARED_FP16_FLASHINFER
                    )
                ):
                    pieces = []
                    for t in range(q_len):
                        piece = self._shared_fp16_decode_triton(
                            q_seq[t : t + 1],
                            kv_cache,
                            attn_metadata.block_table[i : i + 1],
                            cached_len + t + 1,
                            layer,
                        )
                        if piece is None:
                            piece = self._shared_fp16_decode_flashinfer(
                                q_seq[t : t + 1],
                                kv_cache,
                                attn_metadata.block_table[i : i + 1],
                                cached_len + t + 1,
                                layer,
                            )
                        if piece is None:
                            raise RuntimeError(
                                "Shared FP16 paged decode unavailable for "
                                "Gemma4 MTP continuation"
                            )
                        pieces.append(piece)
                    out = torch.cat(pieces, dim=0)
                elif (
                    _SPEC_CONTINUATION_DECODE_FASTPATH
                    and not _TQ_FORCE_CONTINUATION_SDPA
                    and q_len <= _CONTINUATION_DECODE_THRESHOLD
                    and kv_cache.dim() != 5
                ):
                    if q_len == 1:
                        # Single-token continuation is equivalent to decode;
                        # keep the original direct TQ decode path.
                        synth_seq_lens = _arange_cache[
                            cached_len + 1 : seq_len + 1
                        ].contiguous()
                        synth_bt = (
                            attn_metadata.block_table[i : i + 1]
                            .expand(q_len, -1)
                            .contiguous()
                        )
                        out = triton_turboquant_decode_attention(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=synth_bt,
                            seq_lens=synth_seq_lens,
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                            max_num_kv_splits=self.max_num_kv_splits,
                            sliding_window=self._decode_sliding_window,
                        )
                    else:
                        out = self._spec_continuation_decode_attention(
                            q_seq,
                            k_seq,
                            v_seq,
                            kv_cache,
                            attn_metadata.block_table[i : i + 1],
                            cached_len,
                            Pi,
                            centroids,
                            PiT,
                            _arange_cache,
                        )
                        if out is None:
                            out = self._continuation_prefill(
                                layer,
                                q_seq,
                                k_seq,
                                v_seq,
                                kv_cache,
                                attn_metadata.block_table[i : i + 1],
                                cached_len,
                                seq_len,
                                Pi,
                                centroids,
                                force_sdpa=_TQ_FORCE_CONTINUATION_SDPA,
                            )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        force_sdpa=_TQ_FORCE_CONTINUATION_SDPA,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        force_sdpa: bool = False,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        prefix_combine_enabled = (
            _tq_continuation_prefix_combine_enabled(seq_len)
            and not force_sdpa
            and kv_cache.dim() != 5
            and cached_len > 0
            and self._prefill_sliding_window <= 0
            and self._use_flashinfer_for_continuation(q_len)
        )
        if kv_cache.dim() == 5:
            triton_out = self._shared_fp16_decode_triton(
                query,
                kv_cache,
                block_table,
                seq_len,
                layer,
            )
            if triton_out is not None:
                return triton_out

            fi_out = self._shared_fp16_decode_flashinfer(
                query,
                kv_cache,
                block_table,
                seq_len,
                layer,
            )
            if fi_out is not None:
                return fi_out

            # Shared-draft Gemma4 layers can point at a regular FP16 KV cache
            # with layout [blocks, 2, block_size, Hk, D]. Do not feed that
            # cache into the TurboQuant byte dequant kernel.
            block_size = kv_cache.shape[2]
            if cached_len > 0:
                pos = torch.arange(cached_len, device=device, dtype=torch.long)
                page_idx = torch.div(pos, block_size, rounding_mode="floor")
                page_off = pos % block_size
                block_ids = block_table[0, page_idx].long()
                k_cached_trim = kv_cache[block_ids, 0, page_off, :, :]
                v_cached_trim = kv_cache[block_ids, 1, page_off, :, :]
            else:
                k_cached_trim = key_chunk.new_empty((0, Hk, D))
                v_cached_trim = val_chunk.new_empty((0, Hk, D))
            if _GEMMA4_TQ4NC_DEBUG_CONTINUATION and force_sdpa:
                try:
                    pages = max(0, math.ceil(cached_len / block_size))
                    bt = block_table[:, :pages].detach().cpu()
                    logger.warning(
                        "Gemma4 TQ continuation FP16-shared-cache: "
                        "layer=%s shared_target=%s q_len=%s cached_len=%s "
                        "seq_len=%s Hq=%s Hk=%s D=%s block_size=%s pages=%s "
                        "kv_shape=%s kv_stride=%s kv_dtype=%s bt_shape=%s "
                        "bt_min=%s bt_max=%s",
                        getattr(layer, "layer_name", None),
                        getattr(layer, "kv_sharing_target_layer_name", None),
                        q_len,
                        cached_len,
                        seq_len,
                        Hq,
                        Hk,
                        D,
                        block_size,
                        pages,
                        tuple(kv_cache.shape),
                        tuple(kv_cache.stride()),
                        str(kv_cache.dtype),
                        tuple(block_table.shape),
                        int(bt.min().item()) if bt.numel() else -1,
                        int(bt.max().item()) if bt.numel() else -1,
                    )
                except Exception:
                    logger.exception("Gemma4 FP16 shared-cache debug logging failed")
        else:
            block_size = kv_cache.shape[1]
            BLOCK_D = triton.next_power_of_2(D)

            mse_bytes = self._mse_bytes
            val_data_bytes = self._val_data_bytes

            # Dequant cached K/V from TQ cache
            # Allocate slightly over to align to block_size for the grid.
            # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
            alloc_len = math.ceil(cached_len / block_size) * block_size
            buf_shape = (1, Hk, alloc_len, D)
            if prefix_combine_enabled:
                # This experiment only needs the dequantized prefix during the
                # prefix attention call. Use transient buffers so later layer
                # work can reuse the memory once those references are dropped.
                k_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
                v_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
            else:
                # Use WorkspaceManager for dequant buffers.
                # Shared across all layers — saves 60× memory at long context.
                # Required for CUDA Graph capture (per-layer growth incompatible with CG).
                k_buf, v_buf = current_workspace_manager().get_simultaneous(
                    (buf_shape, torch.float16),
                    (buf_shape, torch.float16),
                )
            # Skip .zero_() — kernel writes all positions up to cached_len,
            # and we only read [:cached_len] afterwards.
            k_cached = k_buf[:, :, :alloc_len, :]
            v_cached = v_buf[:, :, :alloc_len, :]

            grid = (alloc_len, 1 * Hk)
            if _GEMMA4_TQ4NC_DEBUG_CONTINUATION and force_sdpa:
                try:
                    pages = max(0, math.ceil(cached_len / block_size))
                    bt = block_table[:, :pages].detach().cpu()
                    bt_min = int(bt.min().item()) if bt.numel() else -1
                    bt_max = int(bt.max().item()) if bt.numel() else -1
                    bt_head = bt.flatten()[:8].tolist()
                    bt_tail = bt.flatten()[-8:].tolist()
                    shared_target = getattr(layer, "kv_sharing_target_layer_name", None)
                    logger.warning(
                        "Gemma4 TQ continuation debug: layer=%s shared_target=%s "
                        "force_sdpa=%s q_len=%s cached_len=%s seq_len=%s "
                        "Hq=%s Hk=%s D=%s block_size=%s alloc_len=%s pages=%s "
                        "kv_shape=%s kv_stride=%s kv_dtype=%s bt_shape=%s "
                        "bt_stride=%s bt_dtype=%s bt_min=%s bt_max=%s "
                        "bt_head=%s bt_tail=%s",
                        getattr(layer, "layer_name", None),
                        shared_target,
                        force_sdpa,
                        q_len,
                        cached_len,
                        seq_len,
                        Hq,
                        Hk,
                        D,
                        block_size,
                        alloc_len,
                        pages,
                        tuple(kv_cache.shape),
                        tuple(kv_cache.stride()),
                        str(kv_cache.dtype),
                        tuple(block_table.shape),
                        tuple(block_table.stride()),
                        str(block_table.dtype),
                        bt_min,
                        bt_max,
                        bt_head,
                        bt_tail,
                    )
                except Exception:
                    logger.exception("Gemma4 TQ continuation debug logging failed")
            _tq_full_dequant_kv[grid](
                kv_cache,
                block_table,
                centroids,
                k_cached,
                v_cached,
                k_cached.stride(0),
                k_cached.stride(1),
                k_cached.stride(2),
                v_cached.stride(0),
                v_cached.stride(1),
                v_cached.stride(2),
                kv_cache.stride(0),
                kv_cache.stride(1),
                kv_cache.stride(2),
                block_table.stride(0),
                HEAD_DIM=D,
                BLOCK_SIZE=block_size,
                NUM_KV_HEADS=Hk,
                MSE_BYTES=mse_bytes,
                KPS=self.tq_config.key_packed_size,
                VQB=self.tq_config.effective_value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                MSE_BITS=self.tq_config.key_mse_bits,
                KEY_FP8=1 if self.tq_config.key_fp8 else 0,
                BLOCK_D=BLOCK_D,
                NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
                FP8_FORMAT=_fp8_format_code(device.index or 0),
                num_warps=4,
            )

            # Inverse-rotate MSE keys back to original space
            if not self.tq_config.key_fp8:
                # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
                Pi_half = layer._tq_Pi_half
                k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
                k_flat = k_flat @ Pi_half
                k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
                    0, 1
                )  # (cached_len, Hk, D) — already fp16
            else:
                k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
                    0, 1
                )  # (cached_len, Hk, D)

            # Skip .contiguous() — the copy into k_full/v_full handles layout
            v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)

        cached_hk = k_cached_trim.shape[1]
        if cached_hk != Hk:
            if cached_hk < Hk and Hk % cached_hk == 0:
                repeat = Hk // cached_hk
                k_cached_trim = k_cached_trim.repeat_interleave(repeat, dim=1)
                v_cached_trim = v_cached_trim.repeat_interleave(repeat, dim=1)
            elif cached_hk > Hk and cached_hk % Hk == 0:
                group = cached_hk // Hk
                k_cached_trim = k_cached_trim.reshape(cached_len, Hk, group, D)[:, :, 0, :]
                v_cached_trim = v_cached_trim.reshape(cached_len, Hk, group, D)[:, :, 0, :]
            else:
                raise RuntimeError(
                    "Unsupported shared KV head mapping: "
                    f"cached_hk={cached_hk}, layer_hk={Hk}, D={D}"
                )

        if prefix_combine_enabled:
            if self._fi_single_qo_indptr_cpu is None:
                self._fi_single_qo_indptr_cpu = torch.empty(
                    2, dtype=torch.int32, pin_memory=True
                )
                self._fi_single_kv_indptr_cpu = torch.empty(
                    2, dtype=torch.int32, pin_memory=True
                )

            self._fi_single_qo_indptr_cpu[0] = 0
            self._fi_single_qo_indptr_cpu[1] = q_len
            self._fi_single_kv_indptr_cpu[0] = 0
            self._fi_single_kv_indptr_cpu[1] = cached_len
            seq_lens_prefix = cached_len * torch.ones(1, dtype=torch.int32)
            seq_lens_q = q_len * torch.ones(1, dtype=torch.int32)
            prefix_plan_key = (
                "continuation_prefix_combine_prefix",
                Hq,
                Hk,
                D,
                str(query.dtype),
                str(k_cached_trim.dtype),
                q_len,
                cached_len,
            )
            prefix_wrapper = self._get_or_plan_flashinfer_prefill_wrapper(
                device,
                prefix_plan_key,
                {
                    "qo_indptr": self._flashinfer_indptr(
                        self._fi_single_qo_indptr_cpu, Hq, D
                    ),
                    "kv_indptr": self._flashinfer_indptr(
                        self._fi_single_kv_indptr_cpu, Hk, D
                    ),
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": False,
                    "window_left": -1,
                    "sm_scale": self.scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": query.dtype,
                    "kv_data_type": k_cached_trim.dtype,
                    "seq_lens": seq_lens_prefix,
                    "seq_lens_q": seq_lens_q,
                    "max_token_per_sequence": q_len,
                    "max_sequence_kv": cached_len,
                },
            )
            prefix_out = torch.empty_like(query)
            prefix_lse = torch.empty(
                (q_len, Hq), dtype=torch.float32, device=device
            )
            prefix_out, prefix_lse = prefix_wrapper.run(
                query,
                k_cached_trim,
                v_cached_trim,
                out=prefix_out,
                lse=prefix_lse,
                return_lse=True,
            )
            del k_cached_trim, v_cached_trim, k_cached, v_cached, k_buf, v_buf

            self._fi_single_kv_indptr_cpu[1] = q_len
            seq_lens_current = q_len * torch.ones(1, dtype=torch.int32)
            current_plan_key = (
                "continuation_prefix_combine_current",
                Hq,
                Hk,
                D,
                str(query.dtype),
                str(key_chunk.dtype),
                q_len,
            )
            current_wrapper = self._get_or_plan_flashinfer_prefill_wrapper(
                device,
                current_plan_key,
                {
                    "qo_indptr": self._flashinfer_indptr(
                        self._fi_single_qo_indptr_cpu, Hq, D
                    ),
                    "kv_indptr": self._flashinfer_indptr(
                        self._fi_single_kv_indptr_cpu, Hk, D
                    ),
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": self._prefill_sliding_window,
                    "sm_scale": self.scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": query.dtype,
                    "kv_data_type": key_chunk.dtype,
                    "seq_lens": seq_lens_current,
                    "seq_lens_q": seq_lens_q,
                    "max_token_per_sequence": q_len,
                    "max_sequence_kv": q_len,
                },
            )
            current_out = torch.empty_like(query)
            current_lse = torch.empty(
                (q_len, Hq), dtype=torch.float32, device=device
            )
            current_out, current_lse = current_wrapper.run(
                query,
                key_chunk,
                val_chunk,
                out=current_out,
                lse=current_lse,
                return_lse=True,
            )
            prefix_lse_for_merge = prefix_lse.transpose(0, 1).contiguous()
            current_lse_for_merge = current_lse.transpose(0, 1).contiguous()
            merge_attn_states(
                current_out,
                prefix_out,
                prefix_lse_for_merge,
                current_out,
                current_lse_for_merge,
            )
            del prefix_out, prefix_lse, current_lse
            del prefix_lse_for_merge, current_lse_for_merge
            logger.info_once(
                "TurboQuant continuation prefix-combine path used: "
                "mode=%s min_tokens=%s seq_len=%s cached_len=%s q_len=%s",
                _TQ_CONTINUATION_PREFIX_COMBINE_MODE,
                _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS,
                seq_len,
                cached_len,
                q_len,
            )
            return current_out

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if self._use_flashinfer_for_continuation(q_len) and not force_sdpa:
            if self._fi_single_qo_indptr_cpu is None:
                self._fi_single_qo_indptr_cpu = torch.empty(
                    2, dtype=torch.int32, pin_memory=True
                )
                self._fi_single_kv_indptr_cpu = torch.empty(
                    2, dtype=torch.int32, pin_memory=True
                )
            self._fi_single_qo_indptr_cpu[0] = 0
            self._fi_single_qo_indptr_cpu[1] = q_len
            self._fi_single_kv_indptr_cpu[0] = 0
            self._fi_single_kv_indptr_cpu[1] = seq_len
            seq_lens = seq_len * torch.ones(1, dtype=torch.int32)
            seq_lens_q = q_len * torch.ones(1, dtype=torch.int32)
            plan_key = (
                "continuation",
                Hq,
                Hk,
                D,
                self._prefill_sliding_window,
                str(query.dtype),
                str(k_full.dtype),
                q_len,
                seq_len,
            )
            prefill_wrapper = self._get_or_plan_flashinfer_prefill_wrapper(
                device,
                plan_key,
                {
                    "qo_indptr": self._flashinfer_indptr(
                        self._fi_single_qo_indptr_cpu, Hq, D
                    ),
                    "kv_indptr": self._flashinfer_indptr(
                        self._fi_single_kv_indptr_cpu, Hk, D
                    ),
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": self._prefill_sliding_window,
                    "sm_scale": self.scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": query.dtype,
                    "kv_data_type": k_full.dtype,
                    "seq_lens": seq_lens,
                    "seq_lens_q": seq_lens_q,
                    "max_token_per_sequence": q_len,
                    "max_sequence_kv": seq_len,
                },
            )
            return prefill_wrapper.run(query, k_full, v_full)

        if self._has_flash_attn_prefill() and not force_sdpa:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q_cpu = torch.zeros(2, dtype=torch.int32, pin_memory=True)
                self._cu_2_k_cpu = torch.zeros(2, dtype=torch.int32, pin_memory=True)
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            self._cu_2_q_cpu[0] = 0
            self._cu_2_q_cpu[1] = q_len
            self._cu_2_k_cpu[0] = 0
            self._cu_2_k_cpu[1] = seq_len
            self._cu_2_q.copy_(self._cu_2_q_cpu, non_blocking=True)
            self._cu_2_k.copy_(self._cu_2_k_cpu, non_blocking=True)
            cu_seqlens_q = self._cu_2_q
            cu_seqlens_k = self._cu_2_k
            return self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            # SDPA fallback: expand KV for GQA, build causal mask
            if force_sdpa and q_len == 1:
                q0 = query[0].float()
                if Hk < Hq:
                    k_attn = k_full.repeat_interleave(self.num_kv_groups, dim=1)
                    v_attn = v_full.repeat_interleave(self.num_kv_groups, dim=1)
                else:
                    k_attn = k_full
                    v_attn = v_full
                scores = (k_attn.float() * q0.unsqueeze(0)).sum(dim=-1)
                probs = torch.softmax(scores.transpose(0, 1) * self.scale, dim=-1)
                out = (v_attn.transpose(0, 1) * probs.unsqueeze(-1)).sum(dim=1)
                return out.unsqueeze(0).to(query.dtype)

            q_chunk = _TQ_CONTINUATION_SDPA_Q_CHUNK
            qk_cap = _TQ_CONTINUATION_SDPA_MAX_QK_CELLS
            if force_sdpa and _TQ_FORCE_DECODE_SDPA_MAX_QK_CELLS > 0:
                qk_cap = (
                    min(qk_cap, _TQ_FORCE_DECODE_SDPA_MAX_QK_CELLS)
                    if qk_cap > 0
                    else _TQ_FORCE_DECODE_SDPA_MAX_QK_CELLS
                )
            if force_sdpa and q_chunk <= 0:
                q_chunk = max(
                    1,
                    min(
                        q_len,
                        qk_cap // max(seq_len, 1) if qk_cap > 0 else q_len,
                    ),
                )
            if q_chunk > 0 and qk_cap > 0:
                capped_q_chunk = max(
                    1,
                    min(q_chunk, qk_cap // max(seq_len, 1)),
                )
                if capped_q_chunk < q_chunk:
                    logger.info_once(
                        "TurboQuant continuation SDPA q-chunk capped: "
                        "requested=%d effective=%d max_qk_cells=%d seq_len=%d",
                        q_chunk,
                        capped_q_chunk,
                        qk_cap,
                        seq_len,
                    )
                    q_chunk = capped_q_chunk
            if 0 < q_chunk < q_len:
                k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
                v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
                k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
                out = torch.empty(q_len, Hq, D, dtype=query.dtype, device=device)
                for q_start in range(0, q_len, q_chunk):
                    q_end = min(q_start + q_chunk, q_len)
                    q_t = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
                    q_pos = (
                        torch.arange(q_start, q_end, device=device).unsqueeze(1)
                        + cached_len
                    )
                    mask = k_pos <= q_pos
                    out_chunk = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        attn_mask=mask,
                        scale=self.scale,
                        enable_gqa=(Hk < Hq),
                    )
                    out[q_start:q_end].copy_(out_chunk[0].transpose(0, 1))
                logger.info_once(
                    "TurboQuant continuation SDPA q-chunk enabled: chunk=%d",
                    q_chunk,
                )
                return out

            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)
            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)
            # Build causal mask: query position p can attend to K position j
            # where j <= cached_len + p (p is 0-indexed within chunk)
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos  # (q_len, seq_len)
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(Hk < Hq),
            )  # (1, Hq, q_len, D)
            return out[0].transpose(0, 1)  # (q_len, Hq, D)

    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _use_shared_draft_decode_sdpa_fallback(
        self,
        layer: torch.nn.Module | None,
    ) -> bool:
        if (
            not _GEMMA4_TQ4NC_SHARED_DRAFT_SDPA_FALLBACK
            or _GEMMA4_TQ4NC_SHARED_DRAFT_NATIVE_DECODE
        ):
            return False
        shared_target = None
        if layer is not None:
            shared_target = getattr(layer, "kv_sharing_target_layer_name", None)
        if shared_target is None:
            shared_target = getattr(self, "kv_sharing_target_layer_name", None)
        return shared_target is not None

    def _use_decode_sdpa_fallback(self) -> bool:
        if _TQ_FORCE_DECODE_SDPA:
            return True
        if _GEMMA4_TQ_DECODE_D256_SDPA_FALLBACK and self.head_size >= 256:
            return True
        return _GEMMA4_TQ_DECODE_D512_SDPA_FALLBACK and self.head_size >= 512

    def _decode_attention_sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        seq_lens = attn_metadata.seq_lens_cpu.tolist()
        for i, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)
            cached_len = max(seq_len - 1, 0)
            output[i : i + 1] = self._continuation_prefill(
                layer,
                query[i : i + 1],
                key[i : i + 1],
                value[i : i + 1],
                kv_cache,
                attn_metadata.block_table[i : i + 1],
                cached_len,
                seq_len,
                Pi,
                centroids,
                force_sdpa=True,
            )
        return output

    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            mid_o_buf, output_buf, lse_buf = current_workspace_manager().get_simultaneous(
                ((B, Hq, S, D + 1), torch.float32),
                ((B, Hq, D), query.dtype),
                ((B, Hq), torch.float32),
            )

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
            sliding_window=self._decode_sliding_window,
        )
        return result
