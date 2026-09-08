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

from collections import OrderedDict
import functools
import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from packaging.version import Version

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up
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
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _fp8_format_code,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

try:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
except ImportError:
    BatchPrefillWithRaggedKVCacheWrapper = None  # type: ignore[assignment]


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
_TQ_CUDAGRAPH_SPEC_DECODE_SAFE = (
    os.getenv("VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE", "0") == "1"
)
_TQ_CUDAGRAPH_SPEC_PREFIX_ROWS = (
    os.getenv("VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS", "0") == "1"
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
    os.getenv("VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE", "auto")
)
_TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS = max(
    0,
    int(
        os.getenv(
            "VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS", "20480"
        )
    ),
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
_TQ_REQUIRE_FLASHINFER_PREFILL = (
    os.getenv("VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL", "0") == "1"
)
_DEFAULT_TQ_FI_PLAN_CACHE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE", "1") == "1"
)
_TQ_FI_PREFILL_PLAN_CACHE_MAXSIZE = max(
    1,
    int(
        os.getenv(
            "VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE_MAXSIZE",
            "16",
        )
    ),
)
_TQ_FI_PREFILL_CUDAGRAPH_SAFE = (
    os.getenv("VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE", "0") == "1"
)
_SM75_TQ_FI_PREFILL_MIN_QUERY_LEN = int(
    os.getenv("VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_QUERY_LEN", "1")
)
_SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN = int(
    os.getenv("VLLM_TURBOQUANT_SM75_FLASHINFER_CONTINUATION_MIN_QUERY_LEN", "1")
)
_TQ_FI_PREFILL_WORKSPACES: dict[tuple[str, str], torch.Tensor] = {}
_TQ_FI_PREFILL_WRAPPERS: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
logger = init_logger(__name__)


def _prepare_tq_flashinfer_prefill_wrapper_cache(cache_key: tuple[Any, ...]):
    """Return a cached wrapper and make room for a new plan when needed.

    CUDA-graph-safe wrappers own indptr buffers referenced by captured graphs;
    without a graph-destruction callback they must remain live for the process
    lifetime. The ordinary path can evict the least-recently-used wrapper so
    long-context plan keys do not consume unbounded GPU workspace.
    """
    wrapper = _TQ_FI_PREFILL_WRAPPERS.get(cache_key)
    if wrapper is not None:
        _TQ_FI_PREFILL_WRAPPERS.move_to_end(cache_key)
        return wrapper

    if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
        logger.warning_once(
            "TurboQuant FlashInfer prefill plan cache eviction is disabled "
            "when CUDA-graph-safe wrappers are enabled; set "
            "VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE=0 to allow "
            "bounded wrapper caching"
        )
    elif len(_TQ_FI_PREFILL_WRAPPERS) >= _TQ_FI_PREFILL_PLAN_CACHE_MAXSIZE:
        # Evict before constructing/planning the replacement. Each wrapper owns
        # an auxiliary GPU workspace, so eviction after planning briefly needs
        # maxsize + 1 workspaces and can defeat the OOM protection.
        _TQ_FI_PREFILL_WRAPPERS.popitem(last=False)
    return None


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


def _get_or_plan_tq_flashinfer_prefill_wrapper(
    device: torch.device,
    plan_key: tuple[Any, ...],
    plan_kwargs: dict[str, Any],
):
    """Plan outside model forward so FlashInfer remains out of CUDA graphs."""
    if BatchPrefillWithRaggedKVCacheWrapper is None:
        return None
    if not _DEFAULT_TQ_FI_PLAN_CACHE:
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(
                envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=device,
            ),
            "NHD",
            backend=_DEFAULT_TQ_FI_BACKEND,
        )
        wrapper.plan(**plan_kwargs)
        return wrapper

    norm_device = _normalize_cuda_device(device)
    cache_key = (str(norm_device), _DEFAULT_TQ_FI_BACKEND, *plan_key)
    wrapper = _prepare_tq_flashinfer_prefill_wrapper_cache(cache_key)
    if wrapper is None:
        workspace = _get_shared_flashinfer_prefill_workspace(
            norm_device, _DEFAULT_TQ_FI_BACKEND
        )
        wrapper_kwargs: dict[str, Any] = {"backend": _DEFAULT_TQ_FI_BACKEND}
        if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
            qo_indptr = plan_kwargs.get("qo_indptr")
            kv_indptr = plan_kwargs.get("kv_indptr")
            if qo_indptr is None or kv_indptr is None:
                raise RuntimeError("FlashInfer cudagraph-safe prefill requires indptr buffers")
            wrapper_kwargs.update(
                {
                    "use_cuda_graph": True,
                    "qo_indptr_buf": torch.empty_like(qo_indptr, device=norm_device),
                    "kv_indptr_buf": torch.empty_like(kv_indptr, device=norm_device),
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
        into a single interleaved slot per head per position. The logical
        (blocks-first, head-major) shape is:

            (num_blocks, num_kv_heads, block_size, slot_size_aligned)

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
        return (num_blocks, num_kv_heads, block_size, tq_config.slot_size_aligned)

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
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests
    # CUDA graph capture uses this graph-safe multi-token continuation path
    # when the scheduler classifies speculative MTP tokens as decode work.
    force_spec_decode: bool = False
    # CPU-resident copies used by the prefill path for per-request iteration
    # without per-step D2H syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    # FlashInfer wrappers are planned by the metadata builder. Keeping plan()
    # out of AttentionImpl.forward avoids leaking a Python/JIT operation into
    # the compiled model or its CUDA graph capture.
    flashinfer_first_chunk_wrapper: Any | None = None
    flashinfer_first_chunk_wrappers: dict[int, Any] | None = None
    flashinfer_continuation_wrappers: dict[int, Any] | None = None
    flashinfer_prefix_combine_wrappers: dict[int, tuple[Any, Any]] | None = None


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    kv_cache_spec: AttentionSpec
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=_TQ_CUDAGRAPH_SPEC_DECODE_SAFE
        )
        self._device = torch.device(device)
        self._flashinfer_prefill_enabled = (
            _DEFAULT_TQ_FI_PREFILL
            and BatchPrefillWithRaggedKVCacheWrapper is not None
            and current_platform.is_cuda()
        )
        if _TQ_REQUIRE_FLASHINFER_PREFILL and not self._flashinfer_prefill_enabled:
            raise RuntimeError(
                "TurboQuant fast route requires the FlashInfer prefill backend; "
                "install a supported FlashInfer build or disable the fast route."
            )
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self._flashinfer_num_qo_heads = model_config.get_num_attention_heads(
            parallel_config
        )
        self._flashinfer_num_kv_heads = kv_cache_spec.num_kv_heads
        self._flashinfer_head_dim = kv_cache_spec.head_size
        self._flashinfer_dtype = model_config.dtype
        self._flashinfer_scale = self._flashinfer_head_dim**-0.5
        self._reserve_workspace()

    def _plan_flashinfer_prefill_wrappers(
        self,
        cam: CommonAttentionMetadata,
        num_decodes: int,
    ) -> tuple[
        Any | None,
        dict[int, Any] | None,
        dict[int, Any] | None,
        dict[int, tuple[Any, Any]] | None,
    ]:
        """Prepare raw-K/V FlashInfer wrappers before model forward."""
        if (
            not self._flashinfer_prefill_enabled
            or cam.max_query_len <= 0
            or cam.query_start_loc_cpu is None
            or cam.seq_lens_cpu_upper_bound is None
        ):
            return None, None, None, None

        qsl = cam.query_start_loc_cpu
        seq_lens = cam.seq_lens_cpu_upper_bound
        q_lens = qsl[1:] - qsl[:-1]
        num_reqs = q_lens.shape[0]
        Hq = self._flashinfer_num_qo_heads
        Hk = self._flashinfer_num_kv_heads
        D = self._flashinfer_head_dim
        dtype = self._flashinfer_dtype
        window_left = -1

        # A complete first chunk can use one batched ragged plan. The explicit
        # CPU check prevents treating a continuation's raw K/V suffix as its
        # entire context.
        if num_decodes == 0 and torch.equal(q_lens, seq_lens[:num_reqs]):
            plan_key = (
                "batch_first_chunk",
                Hq,
                Hk,
                D,
                window_left,
                str(dtype),
                tuple(int(x) for x in qsl.tolist()),
            )
            wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                self._device,
                plan_key,
                {
                    "qo_indptr": qsl,
                    "kv_indptr": qsl,
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": window_left,
                    "sm_scale": self._flashinfer_scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": dtype,
                    "kv_data_type": dtype,
                    "seq_lens": q_lens,
                    "seq_lens_q": q_lens,
                    "max_token_per_sequence": cam.max_query_len,
                    "max_sequence_kv": cam.max_seq_len,
                },
            )
            return wrapper, None, None, None

        # Mixed batches can contain complete first chunks after one or more
        # decode requests. Plan those requests independently; the batched
        # ragged plan above is only valid when every request is first-chunk.
        first_chunk_wrappers: dict[int, Any] = {}
        wrappers: dict[int, Any] = {}
        prefix_combine_wrappers: dict[int, tuple[Any, Any]] = {}
        prefix_combine_supported = (
            getattr(self.kv_cache_spec, "sliding_window", None) is None
        )
        pin_memory = self._device.type == "cuda" and torch.cuda.is_available()
        for request_idx in range(num_decodes, num_reqs):
            q_len = int(q_lens[request_idx])
            seq_len = int(seq_lens[request_idx])
            if q_len == seq_len and q_len > 0:
                qo_indptr = torch.tensor(
                    [0, q_len], dtype=torch.int32, pin_memory=pin_memory
                )
                first_chunk_wrappers[request_idx] = (
                    _get_or_plan_tq_flashinfer_prefill_wrapper(
                        self._device,
                        (
                            "mixed_first_chunk",
                            Hq,
                            Hk,
                            D,
                            window_left,
                            str(dtype),
                            q_len,
                        ),
                        {
                            "qo_indptr": qo_indptr,
                            "kv_indptr": qo_indptr,
                            "num_qo_heads": Hq,
                            "num_kv_heads": Hk,
                            "head_dim_qk": D,
                            "causal": True,
                            "window_left": window_left,
                            "sm_scale": self._flashinfer_scale,
                            "pos_encoding_mode": "NONE",
                            "q_data_type": dtype,
                            "kv_data_type": dtype,
                            "seq_lens": torch.tensor(
                                [q_len], dtype=torch.int32
                            ),
                            "seq_lens_q": torch.tensor(
                                [q_len], dtype=torch.int32
                            ),
                            "max_token_per_sequence": q_len,
                            "max_sequence_kv": q_len,
                        },
                    )
                )
                continue
            if (
                q_len <= _CONTINUATION_DECODE_THRESHOLD
                or q_len < _SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN
                or q_len > seq_len
            ):
                continue
            cached_len = seq_len - q_len
            qo_indptr = torch.tensor(
                [0, q_len], dtype=torch.int32, pin_memory=pin_memory
            )
            if prefix_combine_supported and _tq_continuation_prefix_combine_enabled(
                seq_len
            ):
                prefix_kv_indptr = torch.tensor(
                    [0, cached_len], dtype=torch.int32, pin_memory=pin_memory
                )
                prefix_wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                    self._device,
                    (
                        "continuation_prefix_combine_prefix",
                        Hq,
                        Hk,
                        D,
                        window_left,
                        str(dtype),
                        q_len,
                        cached_len,
                    ),
                    {
                        "qo_indptr": qo_indptr,
                        "kv_indptr": prefix_kv_indptr,
                        "num_qo_heads": Hq,
                        "num_kv_heads": Hk,
                        "head_dim_qk": D,
                        "causal": False,
                        "window_left": window_left,
                        "sm_scale": self._flashinfer_scale,
                        "pos_encoding_mode": "NONE",
                        "q_data_type": dtype,
                        "kv_data_type": torch.float16,
                        "seq_lens": torch.tensor([cached_len], dtype=torch.int32),
                        "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                        "max_token_per_sequence": q_len,
                        "max_sequence_kv": cached_len,
                    },
                )
                current_kv_indptr = torch.tensor(
                    [0, q_len], dtype=torch.int32, pin_memory=pin_memory
                )
                current_wrapper = _get_or_plan_tq_flashinfer_prefill_wrapper(
                    self._device,
                    (
                        "continuation_prefix_combine_current",
                        Hq,
                        Hk,
                        D,
                        window_left,
                        str(dtype),
                        q_len,
                    ),
                    {
                        "qo_indptr": qo_indptr,
                        "kv_indptr": current_kv_indptr,
                        "num_qo_heads": Hq,
                        "num_kv_heads": Hk,
                        "head_dim_qk": D,
                        "causal": True,
                        "window_left": window_left,
                        "sm_scale": self._flashinfer_scale,
                        "pos_encoding_mode": "NONE",
                        "q_data_type": dtype,
                        "kv_data_type": dtype,
                        "seq_lens": torch.tensor([q_len], dtype=torch.int32),
                        "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                        "max_token_per_sequence": q_len,
                        "max_sequence_kv": q_len,
                    },
                )
                if prefix_wrapper is not None and current_wrapper is not None:
                    prefix_combine_wrappers[request_idx] = (
                        prefix_wrapper,
                        current_wrapper,
                    )
                    continue
                if _TQ_REQUIRE_FLASHINFER_PREFILL:
                    raise RuntimeError(
                        "TurboQuant fast route could not plan FlashInfer "
                        "continuation prefix-combine wrappers."
                    )
            kv_indptr = torch.tensor(
                [0, seq_len], dtype=torch.int32, pin_memory=pin_memory
            )
            plan_key = (
                "continuation",
                Hq,
                Hk,
                D,
                window_left,
                str(dtype),
                q_len,
                seq_len,
            )
            wrappers[request_idx] = _get_or_plan_tq_flashinfer_prefill_wrapper(
                self._device,
                plan_key,
                {
                    "qo_indptr": qo_indptr,
                    "kv_indptr": kv_indptr,
                    "num_qo_heads": Hq,
                    "num_kv_heads": Hk,
                    "head_dim_qk": D,
                    "causal": True,
                    "window_left": window_left,
                    "sm_scale": self._flashinfer_scale,
                    "pos_encoding_mode": "NONE",
                    "q_data_type": dtype,
                    "kv_data_type": dtype,
                    "seq_lens": torch.tensor([seq_len], dtype=torch.int32),
                    "seq_lens_q": torch.tensor([q_len], dtype=torch.int32),
                    "max_token_per_sequence": q_len,
                    "max_sequence_kv": seq_len,
                },
            )
        return (
            None,
            first_chunk_wrappers or None,
            wrappers or None,
            prefix_combine_wrappers or None,
        )

    def _reserve_workspace(self) -> None:
        if not is_workspace_manager_initialized():
            return

        scheduler_config = self.vllm_config.scheduler_config
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config

        max_num_reqs = scheduler_config.max_num_seqs
        num_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = self.kv_cache_spec.num_kv_heads
        head_size = self.kv_cache_spec.head_size
        max_num_splits = (
            self.vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        current_workspace_manager().get_simultaneous(
            ((max_num_reqs, num_heads, max_num_splits, head_size + 1), torch.float32),
            ((max_num_reqs, num_heads, head_size), model_config.dtype),
            ((max_num_reqs, num_heads), torch.float32),
        )

        reserve_continuation_prefill = (
            scheduler_config.enable_chunked_prefill
            and scheduler_config.max_num_batched_tokens > _CONTINUATION_DECODE_THRESHOLD
        )
        if not reserve_continuation_prefill:
            return

        max_cached_len = max(0, model_config.max_model_len - 1)
        alloc_len = round_up(max_cached_len, self.kv_cache_spec.block_size)
        cache_buf_shape = (1, num_kv_heads, alloc_len, head_size)
        current_workspace_manager().get_simultaneous(
            (cache_buf_shape, torch.float16),
            (cache_buf_shape, torch.float16),
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        if (
            _TQ_CUDAGRAPH_SPEC_DECODE_SAFE
            and 1 < attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
        ):
            # Capture the MTP continuation through the raw-current-K/V path.
            # Keep the shared warmup sequence length at one: GDN/Mamba reads
            # the same CommonAttentionMetadata, and changing it to q_len
            # makes its SM75 capture address speculative state that was not
            # allocated for the dummy batch. TurboQuant clamps its local
            # prefix length to zero below, so it does not need a synthetic
            # q_len-sized prefix here.
            attn_metadata.force_spec_decode = True
            attn_metadata.seq_lens.fill_(1)
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
        (
            first_chunk_wrapper,
            first_chunk_wrappers,
            continuation_wrappers,
            prefix_combine_wrappers,
        ) = (
            self._plan_flashinfer_prefill_wrappers(cam, num_decodes)
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            flashinfer_first_chunk_wrapper=first_chunk_wrapper,
            flashinfer_first_chunk_wrappers=first_chunk_wrappers,
            flashinfer_continuation_wrappers=continuation_wrappers,
            flashinfer_prefix_combine_wrappers=prefix_combine_wrappers,
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

        self._fi_prefill_workspace: torch.Tensor | None = None
        self._fi_prefill_backend = _DEFAULT_TQ_FI_BACKEND
        self._use_flashinfer_prefill = (
            _DEFAULT_TQ_FI_PREFILL
            and BatchPrefillWithRaggedKVCacheWrapper is not None
            and current_platform.is_cuda()
        )
        self._prefill_sliding_window = -1 if sliding_window is None else int(sliding_window)

        # FlashInfer fa2 is available for the SM75 build. Prefer it for raw
        # K/V prefill and continuation instead of routing those paths to SDPA.
        self.fa_version = (
            None
            if self._use_flashinfer_prefill
            else get_flash_attn_version(head_size=head_size)
        )
        if self._use_flashinfer_prefill:
            capability = current_platform.get_device_capability()
            cap_str = capability.as_version_str() if capability is not None else "unknown"
            logger.info_once(
                "TurboQuant prefill is using FlashInfer backend=%s on CUDA "
                "capability %s (flashinfer=%s).",
                self._fi_prefill_backend,
                cap_str,
                _FLASHINFER_VERSION or "unknown",
            )

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
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
                self._fi_prefill_workspace = torch.empty(
                    envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
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
        wrapper = _prepare_tq_flashinfer_prefill_wrapper_cache(cache_key)
        if wrapper is None:
            workspace = _get_shared_flashinfer_prefill_workspace(
                norm_device, self._fi_prefill_backend
            )
            wrapper_kwargs: dict[str, Any] = {"backend": self._fi_prefill_backend}
            if _TQ_FI_PREFILL_CUDAGRAPH_SAFE:
                qo_indptr = plan_kwargs.get("qo_indptr")
                kv_indptr = plan_kwargs.get("kv_indptr")
                if qo_indptr is None or kv_indptr is None:
                    raise RuntimeError("FlashInfer cudagraph-safe prefill requires indptr buffers")
                wrapper_kwargs.update(
                    {
                        "use_cuda_graph": True,
                        "qo_indptr_buf": torch.empty_like(qo_indptr, device=norm_device),
                        "kv_indptr_buf": torch.empty_like(kv_indptr, device=norm_device),
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
            self._fi_prefill_workspace = _get_shared_flashinfer_prefill_workspace(
                torch.device("cuda", torch.cuda.current_device()),
                self._fi_prefill_backend,
            )

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
        if not current_platform.is_device_capability((7, 5)):
            return True
        return query_len >= _SM75_TQ_FI_PREFILL_MIN_QUERY_LEN

    def _use_flashinfer_for_continuation(self, query_len: int) -> bool:
        if not self._use_flashinfer_prefill:
            return False
        if not current_platform.is_device_capability((7, 5)):
            return True
        return query_len >= _SM75_TQ_FI_CONTINUATION_MIN_QUERY_LEN

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

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        # (B, H, N, C) -> (B, N, H, C) for TQ kernels
        kv_cache = kv_cache.transpose(1, 2)
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

        # (B, H, N, C) -> (B, N, H, C) for TQ kernels
        kv_cache = kv_cache.transpose(1, 2)

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

        if attn_metadata.force_spec_decode:
            attn_out = self._spec_decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT
            )
        elif not attn_metadata.is_prefill:
            # Pure decode batch — fast path
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
            # With spec-as-decode enabled, a uniform MTP continuation has no
            # prefill tail even though its query length is greater than one.
            # Each query needs an incrementing sequence length for causal
            # attention, so it cannot use the regular decode path directly.
            attn_out = self._spec_decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
                query_start_loc_cpu=(
                    attn_metadata.query_start_loc_cpu[: num_decodes + 1]
                    if attn_metadata.query_start_loc_cpu is not None
                    else None
                ),
                seq_lens_cpu=(
                    attn_metadata.seq_lens_cpu[:num_decodes]
                    if attn_metadata.seq_lens_cpu is not None
                    else None
                ),
            )
            if num_decode_tokens > num_decodes:
                # Spec-as-decode admits multiple query tokens per request.
                # Expand request metadata into causal per-query rows before
                # using the decode kernel, just as for a pure MTP batch.
                attn_out[:num_decode_tokens] = self._spec_decode_attention(
                    q[:num_decode_tokens],
                    kv_cache,
                    decode_meta,
                    Pi,
                    centroids,
                    PiT,
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
            # Use the CPU-resident `seq_lens` upper-bound from the metadata
            # (populated in the builder) to compute the prefill sub-batch
            # max without a GPU→CPU sync.
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
                flashinfer_first_chunk_wrappers=(
                    {
                        request_idx - num_decodes: wrapper
                        for request_idx, wrapper in (
                            attn_metadata.flashinfer_first_chunk_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
                flashinfer_continuation_wrappers=(
                    {
                        request_idx - num_decodes: wrapper
                        for request_idx, wrapper in (
                            attn_metadata.flashinfer_continuation_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
                flashinfer_prefix_combine_wrappers=(
                    {
                        request_idx - num_decodes: wrappers
                        for request_idx, wrappers in (
                            attn_metadata.flashinfer_prefix_combine_wrappers or {}
                        ).items()
                        if request_idx >= num_decodes
                    }
                    or None
                ),
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
    ) -> torch.Tensor:
        """Run a multi-token speculative continuation as causal decodes.

        This is the full CUDA-Graph route used by the SM75 fast profile. The
        candidate K/V entries have already been written to the TurboQuant
        cache, so one B=q_len compressed-cache launch preserves the historical
        B=4 graph topology and its fused stage-2 reduction. The raw-K/V prefix
        merge path is intentionally kept out of this route: on SM75 its
        materialized B=4 page-table variant is not graph-safe and regresses the
        validated ~100 tok/s path.
        """
        qsl_cpu = attn_metadata.query_start_loc_cpu
        qsl = (
            qsl_cpu.tolist()
            if qsl_cpu is not None
            else attn_metadata.query_start_loc.tolist()
        )
        num_reqs = attn_metadata.seq_lens.shape[0]
        output = torch.empty_like(query)

        max_seq = max(attn_metadata.max_seq_len, attn_metadata.max_query_len)
        arange_cache: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if arange_cache is None or arange_cache.shape[0] <= max_seq:
            arange_cache = torch.arange(
                max_seq + 1,
                device=query.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
            self._arange_cache = arange_cache

        for request_idx in range(num_reqs):
            q_start = qsl[request_idx]
            q_end = qsl[request_idx + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            rel_seq_lens = arange_cache[1 : q_len + 1]
            seq_lens = (
                attn_metadata.seq_lens[request_idx : request_idx + 1]
                - q_len
                + rel_seq_lens
            ).contiguous()
            block_table = (
                attn_metadata.block_table[request_idx : request_idx + 1]
                .expand(q_len, -1)
                .contiguous()
            )
            output[q_start:q_end] = triton_turboquant_decode_attention(
                query=query[q_start:q_end],
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
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
            ).to(query.dtype)

        return output

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

        # The builder plans FlashInfer before the model forward. The compiled
        # forward only launches the prepared kernel, preserving CUDA-graph
        # capture for decode/spec-decode.
        if attn_metadata.flashinfer_first_chunk_wrapper is not None:
            return attn_metadata.flashinfer_first_chunk_wrapper.run(query, key, value)

        if (
            _TQ_REQUIRE_FLASHINFER_PREFILL
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
            and not attn_metadata.flashinfer_first_chunk_wrappers
        ):
            raise RuntimeError(
                "TurboQuant fast route could not plan FlashInfer prefill; "
                "refusing FlashAttention/SDPA fallback."
            )

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
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
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Prefer the CPU-resident copies from the metadata if populated —
        # otherwise `.tolist()` on GPU tensors forces a synchronizing copy.
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        # Pre-allocate cu_seqlens for single-request flash_attn calls
        # to avoid per-request host→device tensor creation.
        if not hasattr(self, "_cu_2"):
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
                first_chunk_wrapper = (
                    attn_metadata.flashinfer_first_chunk_wrappers.get(i)
                    if attn_metadata.flashinfer_first_chunk_wrappers
                    else None
                )
                if first_chunk_wrapper is not None:
                    out = first_chunk_wrapper.run(q_seq, k_seq, v_seq)
                elif _HAS_FLASH_ATTN:
                    # Assign to slice to avoid gpu/cpu sync.
                    self._cu_2[1:2] = q_len
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
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # The current MTP chunk is already present in the
                    # compressed cache, but reading it back changes the
                    # speculative logits.  Keep the raw current K/V and
                    # merge it with attention over the compressed prefix.
                    if _SPEC_CONTINUATION_DECODE_FASTPATH and q_len > 1:
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
                    else:
                        out = None

                    if out is None:
                        # Fast path: treat each query as a decode request with
                        # incremental seq_lens for the compressed-cache path.
                        synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                        synth_bt = attn_metadata.block_table[i : i + 1].expand(
                            q_len, -1
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
                        )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    if (
                        _TQ_REQUIRE_FLASHINFER_PREFILL
                        and (
                            (
                                attn_metadata.flashinfer_continuation_wrappers is None
                                or i
                                not in attn_metadata.flashinfer_continuation_wrappers
                            )
                            and (
                                attn_metadata.flashinfer_prefix_combine_wrappers
                                is None
                                or i
                                not in attn_metadata.flashinfer_prefix_combine_wrappers
                            )
                        )
                    ):
                        raise RuntimeError(
                            "TurboQuant fast route could not plan FlashInfer "
                            "continuation prefill; refusing FlashAttention/SDPA "
                            "fallback."
                        )
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
                        flashinfer_wrapper=(
                            attn_metadata.flashinfer_continuation_wrappers.get(i)
                            if attn_metadata.flashinfer_continuation_wrappers
                            else None
                        ),
                        flashinfer_prefix_combine_wrappers=(
                            attn_metadata.flashinfer_prefix_combine_wrappers.get(i)
                            if attn_metadata.flashinfer_prefix_combine_wrappers
                            else None
                        ),
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
        cached_len: int | torch.Tensor,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None,
        arange_cache: torch.Tensor,
    ) -> torch.Tensor | None:
        """Merge compressed-prefix attention with raw MTP continuation KV.

        The cache update runs before attention, so reading the current
        speculative tokens from the quantized cache is lossy.  Computing the
        prefix once with the TQ kernel and the tiny current chunk with PyTorch
        preserves MTP quality without falling back to full prefix dequant.
        """
        if isinstance(cached_len, int) and cached_len <= 0:
            return None

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        if Hk <= 0 or Hq % Hk != 0:
            return None

        if isinstance(cached_len, int):
            prefix_seq_lens = torch.full(
                (q_len,),
                cached_len,
                dtype=arange_cache.dtype,
                device=query.device,
            )
            has_prefix = cached_len > 0
        else:
            # The capture warmup has no cached prefix. Keep that zero length:
            # its block table does not name a valid TQ page yet, so forcing a
            # one-token TQ read would address the sentinel block entry. The
            # Triton decode stage skips empty splits, and the graph-local
            # masks below turn its unused output into the exact empty-prefix
            # state. Replay with a positive cached_len follows the same graph
            # topology and reads the real compressed prefix.
            prefix_seq_lens = cached_len.reshape(1).expand(q_len)
            has_prefix = cached_len > 0
        prefix_lse = torch.empty(q_len, Hq, dtype=torch.float32, device=query.device)
        prefix_out = torch.empty_like(query)
        if _TQ_CUDAGRAPH_SPEC_PREFIX_ROWS and not isinstance(cached_len, int):
            # All MTP candidates attend to the same compressed prefix. On
            # SM75, capture the fixed-width candidates as individual B=1 TQ
            # decodes to avoid the B=4 materialized page-table path. This is
            # still fully captured and replayed by CUDA Graph; only the graph
            # topology differs.
            prefix_seq_len = cached_len.reshape(1)
            for row in range(q_len):
                triton_turboquant_decode_attention(
                    query=query[row : row + 1],
                    kv_cache=kv_cache,
                    block_table=block_table,
                    seq_lens=prefix_seq_len,
                    Pi=Pi,
                    centroids=centroids,
                    scale=self.scale,
                    mse_bits=self.tq_config.key_mse_bits,
                    key_packed_size=self.tq_config.key_packed_size,
                    value_quant_bits=self.tq_config.effective_value_quant_bits,
                    key_fp8=self.tq_config.key_fp8,
                    norm_correction=self.tq_config.norm_correction,
                    PiT=PiT,
                    output_buf=prefix_out[row : row + 1],
                    lse_buf=prefix_lse[row : row + 1],
                    max_num_kv_splits=self.max_num_kv_splits,
                )
        else:
            prefix_bt = block_table.expand(q_len, -1).contiguous()
            triton_turboquant_decode_attention(
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
                output_buf=prefix_out,
                lse_buf=prefix_lse,
                max_num_kv_splits=self.max_num_kv_splits,
            )
        if not isinstance(has_prefix, bool):
            prefix_out = torch.where(
                has_prefix.reshape(1, 1, 1),
                prefix_out,
                torch.zeros_like(prefix_out),
            )
            prefix_lse = torch.where(
                has_prefix.reshape(1, 1),
                prefix_lse,
                torch.full_like(prefix_lse, float("-inf")),
            )

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
        flashinfer_wrapper: Any | None = None,
        flashinfer_prefix_combine_wrappers: tuple[Any, Any] | None = None,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid.
        # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, Hk, alloc_len, D)
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

        if flashinfer_prefix_combine_wrappers is not None:
            prefix_wrapper, current_wrapper = flashinfer_prefix_combine_wrappers
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
            merged_out = torch.empty_like(query)
            merge_attn_states(
                merged_out,
                prefix_out,
                prefix_lse.transpose(0, 1).contiguous(),
                current_out,
                current_lse.transpose(0, 1).contiguous(),
            )
            logger.info_once(
                "TurboQuant continuation prefix-combine path used: "
                "mode=%s min_tokens=%s seq_len=%s cached_len=%s q_len=%s",
                _TQ_CONTINUATION_PREFIX_COMBINE_MODE,
                _TQ_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS,
                seq_len,
                cached_len,
                q_len,
            )
            return merged_out

        # Concatenate cached + current chunk K/V (match query dtype)
        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
        qdtype = query.dtype
        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        if flashinfer_wrapper is not None:
            return flashinfer_wrapper.run(query, k_full, v_full)

        # Attention: q_len queries attending to seq_len K/V with causal mask
        if _HAS_FLASH_ATTN:
            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
            if not hasattr(self, "_cu_2_q"):
                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
            # Assigning to slice uses fill_ which avoids cpu/gpu sync.
            self._cu_2_q[1:2] = q_len
            self._cu_2_k[1:2] = seq_len
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
        # Acquire shared decode scratch buffers from WorkspaceManager.
        # Layers execute sequentially so one set of buffers is sufficient.
        # Falls back to kernel-internal allocation if workspace unavailable.
        B = query.shape[0]
        D = self.head_size
        S = self.max_num_kv_splits
        Hq = self.num_heads
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
            mid_o_buf, output_buf, lse_buf = (
                current_workspace_manager().get_simultaneous(
                    ((B, Hq, S, D + 1), torch.float32),
                    ((B, Hq, D), query.dtype),
                    ((B, Hq), torch.float32),
                )
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
        )
        return result
