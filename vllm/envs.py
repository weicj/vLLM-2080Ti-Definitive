# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_PORT: int | None = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: str | None = None
    LD_LIBRARY_PATH: str | None = None
    VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE: int = 256
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: str | None = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60
    VLLM_ENGINE_READY_TIMEOUT_S: int = 600
    VLLM_API_KEY: str | None = None
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE: bool = False
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_ENDPOINT_URL: str | None = None
    VLLM_MODEL_REDIRECT_PATH: str | None = None
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
    VLLM_NO_USAGE_STATS: bool = False
    VLLM_DO_NOT_TRACK: bool = False
    VLLM_USAGE_SOURCE: str = "production"
    VLLM_CONFIGURE_LOGGING: bool = True
    VLLM_LOGGING_LEVEL: str = "INFO"
    VLLM_LOGGING_PREFIX: str = ""
    VLLM_LOGGING_STREAM: str = "ext://sys.stdout"
    VLLM_LOGGING_CONFIG_PATH: str | None = None
    VLLM_LOGGING_COLOR: str = "auto"
    NO_COLOR: bool = False
    VLLM_LOG_STATS_INTERVAL: float = 10.0
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_USE_FLASHINFER_SAMPLER: bool = True
    VLLM_SM75_SPEC_SYNC_MODE: Literal["auto", "safe", "nosync"] = "auto"
    VLLM_INT8KV_FA_PREFILL: bool = False
    VLLM_INT8KV_FLASHINFER_PREFILL_BACKEND: str = "fa2"
    VLLM_INT8KV_FA_RAGGED_PREFILL: bool = True
    VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT: bool = False
    VLLM_INT8KV_FA_CONTINUATION_DEQUANT: bool = False
    VLLM_INT8KV_FA_CONTINUATION_MAX_TOKENS: int = 65536
    VLLM_INT8KV_FA_CONTINUATION_MIN_Q: int = 128
    VLLM_INT8KV_FA_CASCADE_DEQUANT: bool = False
    VLLM_INT8KV_FA_CASCADE_TILE_TOKENS: int = 65536
    VLLM_INT8KV_FA_DIRECT_PAGED: bool = False
    VLLM_INT8KV_ALIGNED_HEAD_STRIDE: bool = False
    VLLM_INT8KV_DEBUG_VERIFY: bool = False
    VLLM_INT8KV_DEBUG_COMPARE: bool = False
    VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH: bool = False
    VLLM_TURBOQUANT_FLASHINFER_BACKEND: str = "fa2"
    VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL: bool = True
    VLLM_TURBOQUANT_SM75_FLASHINFER_PREFILL_MIN_HEAD_DIM: int = 1024
    VLLM_TURBOQUANT_FLASHINFER_PREFILL_PLAN_CACHE: bool = True
    VLLM_TURBOQUANT_FLASHINFER_PREFILL_CUDAGRAPH_SAFE: bool = False
    VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE: Literal["off", "on", "auto"] = "auto"
    VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS: int = 20480
    VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS: int = 0
    VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK: int = 0
    VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS: int = 0
    VLLM_TURBOQUANT_FORCE_DECODE_SDPA: bool = False
    VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA: bool = False
    VLLM_TURBOQUANT_FORCE_DECODE_SDPA_MAX_QK_CELLS: int = 131072
    VLLM_TURBOQUANT_MAX_KV_SPLITS: int | None = None
    VLLM_TURBOQUANT_DECODE_BLOCK_KV: int = 2
    VLLM_TURBOQUANT_K8V4_FP8_FORMAT: str = "auto"
    VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE: bool = False
    VLLM_TURBOQUANT_SKIP_PREFILL_STORE: bool = False
    VLLM_PP_LAYER_PARTITION: str | None = None
    VLLM_CPU_KVCACHE_SPACE: int | None = 0
    VLLM_CPU_OMP_THREADS_BIND: str = "auto"
    VLLM_CPU_NUM_OF_RESERVED_CPU: int | None = None
    VLLM_CPU_SGL_KERNEL: bool = False
    VLLM_CPU_ATTN_SPLIT_KV: bool = True
    VLLM_ZENTORCH_WEIGHT_PREPACK: bool = True
    VLLM_CPU_INT4_W4A8: bool = True
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: int = 512
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: Literal["auto", "nccl", "shm"] = "auto"
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_USE_RAY_WRAPPED_PP_COMM: bool = True
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND: bool = False
    VLLM_XLA_USE_SPMD: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "fork"
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")
    VLLM_ASSETS_CACHE_MODEL_CLEAN: bool = False
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_MEDIA_CACHE: str = ""
    VLLM_MEDIA_CACHE_MAX_SIZE_MB: int = 5120
    VLLM_MEDIA_CACHE_TTL_HOURS: float = 24
    VLLM_MEDIA_FETCH_MAX_RETRIES: int = 3
    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool = True
    VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
    VLLM_MAX_AUDIO_CLIP_FILESIZE_MB: int = 25
    VLLM_VIDEO_LOADER_BACKEND: str = "opencv"
    VLLM_MEDIA_CONNECTOR: str = "http"
    VLLM_MM_HASHER_ALGORITHM: str = "blake3"
    VLLM_TARGET_DEVICE: str = "cuda"
    VLLM_MAIN_CUDA_VERSION: str = "13.0"
    VLLM_FLOAT32_MATMUL_PRECISION: Literal["highest", "high", "medium"] = "highest"
    VLLM_BATCH_INVARIANT: bool = False
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX: bool = False
    VLLM_DOCKER_BUILD_CONTEXT: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    VLLM_RPC_TIMEOUT: int = 10000  # ms
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5  # seconds
    VLLM_MAX_N_SEQUENCES: int = 16384
    VLLM_PLUGINS: list[str] | None = None
    VLLM_LORA_RESOLVER_CACHE_DIR: str | None = None
    VLLM_LORA_RESOLVER_HF_REPO_LIST: str | None = None
    VLLM_USE_AOT_COMPILE: bool = False
    VLLM_USE_BYTECODE_HOOK: bool = True
    VLLM_FORCE_AOT_LOAD: bool = False
    VLLM_USE_MEGA_AOT_ARTIFACT: bool = False
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE: bool = True
    VLLM_DISABLE_PYNCCL: bool = False
    VLLM_USE_OINK_OPS: bool = False
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_AITER_MHA: bool = True
    VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: bool = False
    VLLM_ROCM_USE_AITER_TRITON_ROPE: bool = False
    VLLM_ROCM_USE_AITER_FP8BMM: bool = True
    VLLM_ROCM_USE_AITER_FP4BMM: bool = True
    VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: bool = False
    VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: bool = False
    VLLM_ROCM_USE_AITER_TRITON_GEMM: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT: bool = False
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    VLLM_USE_LAYERNAME: bool = True
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ""
    VLLM_CUDART_SO_PATH: str | None = None
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_USE_STANDALONE_COMPILE: bool = True
    VLLM_ENABLE_PREGRAD_PASSES: bool = True
    VLLM_DP_MASTER_IP: str = ""
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_RANDOMIZE_DP_DUMMY_INPUTS: bool = False
    VLLM_RAY_DP_PACK_STRATEGY: Literal["strict", "fill", "span"] = "strict"
    VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY: str = ""
    VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: str = ""
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_MARLIN_INPUT_DTYPE: Literal["int8", "fp8"] | None = None
    VLLM_HUMMING_ONLINE_QUANT_CONFIG: dict[str, Any] | None = None
    VLLM_HUMMING_INPUT_QUANT_CONFIG: dict[str, Any] | None = None
    VLLM_HUMMING_USE_F16_ACCUM: bool = False
    VLLM_HUMMING_MOE_GEMM_TYPE: Literal["indexed", "grouped", "auto"] | None = None
    VLLM_MXFP4_USE_MARLIN: bool | None = None
    VLLM_DEEPEPLL_NVFP4_DISPATCH: bool = False
    VLLM_V1_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: int | None = None
    VLLM_TPU_USING_PATHWAYS: bool = False
    VLLM_USE_DEEP_GEMM: bool = True
    VLLM_MOE_USE_DEEP_GEMM: bool = True
    VLLM_USE_DEEP_GEMM_E8M0: bool = True
    VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES: bool = True
    VLLM_DEEP_GEMM_WARMUP: Literal[
        "skip",
        "full",
        "relax",
    ] = "relax"
    VLLM_USE_FUSED_MOE_GROUPED_TOPK: bool = True
    VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER: bool = True
    VLLM_USE_FLASHINFER_MOE_FP16: bool = False
    VLLM_USE_FLASHINFER_MOE_FP8: bool = False
    VLLM_USE_FLASHINFER_MOE_FP4: bool = False
    VLLM_USE_FLASHINFER_MOE_INT4: bool = False
    VLLM_FLASHINFER_MOE_BACKEND: Literal["throughput", "latency", "masked_gemm"] = (
        "latency"
    )
    VLLM_FLASHINFER_ALLREDUCE_BACKEND: Literal["auto", "trtllm", "mnnvl"] = "auto"
    VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256
    VLLM_ALLOW_INSECURE_SERIALIZATION: bool = False
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_MOONCAKE_BOOTSTRAP_PORT: int = 8998
    VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE: int = 163840
    VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS: int = 1
    VLLM_MQ_MAX_CHUNK_BYTES_MB: int = 16
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = 300
    VLLM_KV_CACHE_LAYOUT: Literal["NHD", "HND"] | None = None
    VLLM_SSM_CONV_STATE_LAYOUT: Literal["SD", "DS"] | None = None
    VLLM_COMPUTE_NANS_IN_LOGITS: bool = False
    VLLM_USE_NVFP4_CT_EMULATIONS: bool = False
    VLLM_ROCM_QUICK_REDUCE_QUANTIZATION: Literal[
        "FP", "INT8", "INT6", "INT4", "NONE"
    ] = "NONE"
    VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16: bool = True
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 480
    VLLM_MORIIO_CONNECTOR_READ_MODE: bool = False
    VLLM_MORIIO_QP_PER_TRANSFER: int = 1
    VLLM_MORIIO_POST_BATCH_SIZE: int = -1
    VLLM_MORIIO_NUM_WORKERS: int = 1
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: int = 480
    VLLM_ENABLE_CUDAGRAPH_GC: bool = False
    VLLM_LOOPBACK_IP: str = ""
    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True
    VLLM_ENABLE_RESPONSES_API_STORE: bool = False
    VLLM_NVFP4_GEMM_BACKEND: str | None = None
    VLLM_HAS_FLASHINFER_CUBIN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_BF16: bool = False
    VLLM_ROCM_FP8_MFMA_PAGE_ATTN: bool = False
    VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS: bool = False
    VLLM_ALLREDUCE_USE_SYMM_MEM: bool = True
    VLLM_ALLREDUCE_USE_FLASHINFER: bool = False
    VLLM_TUNED_CONFIG_FOLDER: str | None = None
    VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS: set[str] = set()
    VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT: bool = False
    VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS: bool = False
    VLLM_SYSTEM_START_DATE: str | None = None
    VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY: bool = False
    VLLM_ENFORCE_STRICT_TOOL_CALLING: bool = True
    VLLM_CUSTOM_SCOPES_FOR_PROFILING: bool = False
    VLLM_NVTX_SCOPES_FOR_PROFILING: bool = False
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = True
    VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME: str = "VLLM_OBJECT_STORAGE_SHM_BUFFER"
    VLLM_DEEPEP_BUFFER_SIZE_MB: int = 1024
    VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE: bool = False
    VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL: bool = False
    VLLM_DBO_COMM_SMS: int = 20
    VLLM_PATTERN_MATCH_DEBUG: str | None = None
    VLLM_DEBUG_DUMP_PATH: str | None = None
    VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE: bool = True
    VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING: bool = True
    VLLM_USE_NCCL_SYMM_MEM: bool = False
    VLLM_NCCL_INCLUDE_PATH: str | None = None
    VLLM_USE_FBGEMM: bool = False
    VLLM_GC_DEBUG: str = ""
    VLLM_DEBUG_WORKSPACE: bool = False
    VLLM_DISABLE_SHARED_EXPERTS_STREAM: bool = False
    VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD: int = 256
    VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: int = 1024
    VLLM_COMPILE_CACHE_SAVE_FORMAT: Literal["binary", "unpacked"] = "binary"
    VLLM_USE_V2_MODEL_RUNNER: bool = False
    VLLM_LOG_MODEL_INSPECTION: bool = False
    VLLM_DEBUG_MFU_METRICS: bool = False
    VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY: bool = False
    VLLM_WEIGHT_OFFLOADING_DISABLE_UVA: bool = False
    VLLM_DISABLE_LOG_LOGO: bool = False
    VLLM_LORA_DISABLE_PDL: bool = False
    VLLM_ENABLE_CUDA_COMPATIBILITY: bool = False
    VLLM_CUDA_COMPATIBILITY_PATH: str | None = None
    VLLM_SKIP_MODEL_NAME_VALIDATION: bool = False
    """If set, vLLM will skip model name validation in API requests.
    This allows any model name to be accepted in the 'model' field of requests,
    making the server model-name agnostic. Useful for proxy/gateway scenarios."""
    VLLM_ELASTIC_EP_SCALE_UP_LAUNCH: bool = False
    VLLM_ELASTIC_EP_DRAIN_REQUESTS: bool = False
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = True
    VLLM_NIXL_EP_MAX_NUM_RANKS: int = 32
    VLLM_XPU_ENABLE_XPU_GRAPH: bool = False
    VLLM_XPU_USE_SAMPLER_KERNEL: bool = True
    VLLM_LORA_ENABLE_DUAL_STREAM: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_json_str_or_file(value: str | None) -> dict[str, Any]Û~}ÞÚ$z{-®éÜj×æBõö–×÷'Eõò‚'F÷&6‚"’çfW'6–öâæ†——2æ÷BæöæPÐ¢VÇ6R##"ÀÐ¢Ð¢’ÀÐ¢2Væ&ÆRÖ…öWF÷GVæRb6ö÷&F–æFUöFW66VçE÷GVæ–ær–â–æGV7F÷%ö6öæf–pÐ¢2Fò6ö×–ÆR7FF–26†W276VBg&öÒ6ö×–ÆU÷6—¦W2–â6ö×–ÆF–öåö6öæf–pÐ¢2–b6WBFòÂVæ&ÆRÖ…öWF÷GVæS²'’FVfVÇBÂF†—2—2Væ&ÆVBƒÐ¢%dÄÄÕôTä$ÄUô”äET5Dõ%ôÔ…ôUDõETäR#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôTä$ÄUô”äET5Dõ%ôÔ…ôUDõETäR"Â#"’Ð¢’ÀÐ¢2–b6WBFòÂVæ&ÆR6ö÷&F–æFUöFW66VçE÷GVæ–æs°Ð¢2'’FVfVÇBÂF†—2—2Væ&ÆVBƒÐ¢%dÄÄÕôTä$ÄUô”äET5Dõ%ô4ôõ$D”äDUôDU44TåEõETä”är#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôTä$ÄUô”äET5Dõ%ô4ôõ$D”äDUôDU44TåEõETä”är"Â#"’Ð¢’ÀÐ¢2fÆrFòVæ&ÆRä44Â7–ÖÖWG&–2ÖVÖ÷'’ÆÆö6F–öâæB&Vv—7G&F–öàÐ¢%dÄÄÕõU4Uôä44Åõ5”ÔÕôÔTÒ#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõU4Uôä44Åõ5”ÔÕôÔTÒ"Â#"’Ð¢’ÀÐ¢2ä44Â†VFW"F€Ð¢%dÄÄÕôä44Åô”ä4ÅTDUõD‚#¢ÆÖ&F¢÷2æVçf—&öâævWB‚%dÄÄÕôä44Åô”ä4ÅTDUõD‚"ÂæöæR’ÀÐ¢2fÆrFòVæ&ÆRd$vVÖÒ¶W&æVÇ2öâÖöFVÂW†V7WF–öàÐ¢%dÄÄÕõU4Uôd$tTÔÒ#¢ÆÖ&F¢&ööÂ†–çB†÷2ævWFVçb‚%dÄÄÕõU4Uôd$tTÔÒ"Â#"’’’ÀÐ¢2t2FV'Vr6öæf–pÐ¢2ÒdÄÄÕôt5ôDT%TsÓ¢F—6&ÆRt2FV'VvvW Ð¢2ÒdÄÄÕôt5ôDT%TsÓ¢Væ&ÆRt2FV'VvvW"v—F‚v2æ6öÆÆV7BVÇ6VBF–ÖW0Ð¢2ÒdÄÄÕôt5ôDT%TsÒw²'F÷öö&¦V7G2#£WÒs¢Væ&ÆRt2FV'VvvW"v—F€Ð¢2F÷R6öÆÆV7FVBö&¦V7G0Ð¢%dÄÄÕôt5ôDT%Tr#¢ÆÖ&F¢÷2ævWFVçb‚%dÄÄÕôt5ôDT%Tr"Â""’ÀÐ¢2FV'Vrv÷&·76RÆÆö6F–öç2àÐ¢2Æövv–æröbv÷&·76R&W6—¦R÷W&F–öç2àÐ¢%dÄÄÕôDT%Tuõtõ$µ54R#¢ÆÖ&F¢&ööÂ†–çB†÷2ævWFVçb‚%dÄÄÕôDT%Tuõtõ$µ54R"Â#"’’’ÀÐ¢2F—6&ÆW2&ÆÆVÂW†V7WF–öâöb6†&VEöW‡W'G2f–6W&FR7VF7G&VÐÐ¢%dÄÄÕôD•4$ÄUõ4„$TEôU…U%E5õ5E$TÒ#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôD•4$ÄUõ4„$TEôU…U%E5õ5E$TÒ"Â#"’Ð¢’ÀÐ¢2Æ–Ö—G2v†VâvR'Vâ6†&VEöW‡W'G2–â6W&FR7G&VÒàÐ¢2vRf÷VæB÷WBF†Bf÷"Æ&vR&F6‚6—¦W2ÂF†R6W&FR7G&VÐÐ¢2W†V7WF–öâ—2æ÷B&VæVf–6–Â†Ö÷7BÆ–¶VÇ’&V6W6RöbF†R–çWB6ÆöæRÐ¢2DôDò†ÆW†Ò×&VF†B“¢GVæRFò&RÖ÷&RG–æÖ–2&6VBöâuRG—PÐ¢%dÄÄÕõ4„$TEôU…U%E5õ5E$TÕõDô´TåõD…$U4„ôÄB#¢ÆÖ&F¢–çB€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõ4„$TEôU…U%E5õ5E$TÕõDô´TåõD…$U4„ôÄB"Â#Sb’Ð¢’ÀÐ¢2Fö¶VâÖ6÷VçB7WFöfbf÷"×VÇF’×7G&VÒ÷fW&ÆöbF†RGFVçF–öâ–çW@Ð¢2tTÔÒv—F‚W†–Æ–'’tTÔ×2†RærâgW6VE÷w÷v·b÷fW&ÆVBv—F‚–æFW†W Ð¢2vV–v‡G2ò·b×66÷&R&ö¦V7F–öç2–âFVW6VV²ÕcB’âB÷"&VÆ÷rF†—2ÖçÐ¢2Fö¶Vç2F†Re‚Ö–âtTÔÒ†2–FÆR4×2Fò6†&Rv—F‚F†R&cbW‚tTÔ×0Ð¢2æB÷fW&Æ—2RÓCRRv–ã²&÷fR—BF†Re‚tTÔÒ6GW&FW2F†RFWf–6PÐ¢2æBF†R7&÷72×7G&VÒ7–æ2&V6öÖW2W&R÷fW&†VBâ6WBFòFòF—6&ÆPÐ¢2F†R×VÇF’×7G&VÒF‚VçF—&VÇ’â6VR5"CS#bf÷"F†RV×—&–6Â&W7VÇ@Ð¢2f÷"F†RFVfVÇBfÇVRöb#BFö¶Vç2àÐ¢%dÄÄÕôÕTÅD•õ5E$TÕôtTÔÕõDô´TåõD…$U4„ôÄB#¢ÆÖ&F¢–çB€Ð¢÷2ævWFVçb‚%dÄÄÕôÕTÅD•õ5E$TÕôtTÔÕõDô´TåõD…$U4„ôÄB"Â##B"Ð¢’ÀÐ¢2f÷&ÖBf÷"6f–ærF÷&6‚æ6ö×–ÆR66†R'F–f7G0Ð¢2Ò&&–æ'’#¢6fW22&–æ'’f–ÆPÐ¢26fRf÷"×VÇF—ÆRfÆÆÒ6W'fR&ö6W76W266W76–ærF†R6ÖRF÷&6‚6ö×–ÆR66†RàÐ¢2Ò'Vç6¶VB#¢6fW22F—&V7F÷'’7G'V7GW&R†f÷"–ç7V7F–öâöFV'Vvv–ærÐ¢2äõB×VÇF—&ö6W726fRÒ&6R6öæF—F–öç2Ö’ö67W"v—F‚×VÇF—ÆR&ö6W76W2àÐ¢2ÆÆ÷w2f–Wv–æræB6WGF–ær'&V·ö–çG2–â–æGV7F÷"w26öFR÷WGWBf–ÆW2àÐ¢%dÄÄÕô4ôÕ”ÄUô44„Uõ4dUôdõ$ÔB#¢Vçe÷v—F…ö6†ö–6W2€Ð¢%dÄÄÕô4ôÕ”ÄUô44„Uõ4dUôdõ$ÔB"Â&&–æ'’"Â²&&–æ'’"Â'Vç6¶VB%ÐÐ¢’ÀÐ¢2fÆrFòVæ&ÆRc"ÖöFVÂ'VææW"àÐ¢%dÄÄÕõU4Uõc%ôÔôDTÅõ%TääU"#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõU4Uõc%ôÔôDTÅõ%TääU""Â#"’Ð¢’ÀÐ¢2ÆörÖöFVÂ–ç7V7F–öâgFW"ÆöF–æràÐ¢2–bVæ&ÆVBÂÆöw2G&ç6f÷&ÖW'2×7G–ÆR†–W&&6†–6Âf–WröbF†RÖöFVÀÐ¢2v—F‚VçF—¦F–öâÖWF†öG2æBGFVçF–öâ&6¶VæG2àÐ¢%dÄÄÕôÄôuôÔôDTÅô”å5T5D”ôâ#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôÄôuôÔôDTÅô”å5T5D”ôâ"Â#"’Ð¢’ÀÐ¢2FV'VrÆövv–ærf÷"ÒÖVæ&ÆRÖÖgRÖÖWG&–70Ð¢%dÄÄÕôDT%TuôÔeUôÔUE$”52#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôDT%TuôÔeUôÔUE$”52"Â#"’Ð¢’ÀÐ¢2F—6&ÆRW6–ær—F÷&6‚w2–âÖVÖ÷'’f÷"5RöffÆöF–æràÐ¢%dÄÄÕõtT”t…EôôddÄôD”äuôD•4$ÄUõ”åôÔTÔõ%’#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõtT”t…EôôddÄôD”äuôD•4$ÄUõ”åôÔTÔõ%’"Â#"’Ð¢’ÀÐ¢2F—6&ÆRW6–ærUd…Væ–f–VBf—'GVÂFG&W76–ær’f÷"5RöffÆöF–æràÐ¢%dÄÄÕõtT”t…EôôddÄôD”äuôD•4$ÄUõUd#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõtT”t…EôôddÄôD”äuôD•4$ÄUõUd"Â#"’Ð¢’ÀÐ¢2F—6&ÆRÆövv–æröbdÄÄÒÆövòB6W'fW"7F'GWF–ÖRàÐ¢%dÄÄÕôD•4$ÄUôÄôuôÄôtò#¢ÆÖ&F¢&ööÂ†–çB†÷2ævWFVçb‚%dÄÄÕôD•4$ÄUôÄôuôÄôtò"Â#"’’’ÀÐ¢2F—6&ÆRDÂf÷"Æõ$Â2Væ&Æ–ærDÂv—F‚Æõ$öâ4Ó6W6W0Ð¢2G&—Föâ6ö×–ÆF–öâFòf–ÂàÐ¢%dÄÄÕôÄõ$ôD•4$ÄUõDÂ#¢ÆÖ&F¢&ööÂ†–çB†÷2ævWFVçb‚%dÄÄÕôÄõ$ôD•4$ÄUõDÂ"Â#"’’’ÀÐ¢2Væ&ÆR5TD6ö×F–&–Æ—G’ÖöFRf÷"FF6VçFW"uW2v—F‚öÆFW Ð¢2G&—fW"fW'6–öç2F†âF†R5TDFööÆ¶—BÖ¦÷"fW'6–öâöbdÄÄÒàÐ¢%dÄÄÕôTä$ÄUô5TDô4ôÕD”$”Ä•E’#¢ÆÖ&F¢€Ð¢÷2æVçf—&öâævWB‚%dÄÄÕôTä$ÄUô5TDô4ôÕD”$”Ä•E’"Â#"’ç7G&—‚’æÆ÷vW"‚Ð¢–â‚#"Â'G'VR"Ð¢’ÀÐ¢2F‚FòF†R5TD6ö×F–&–Æ—G’Æ–'&&–W2v†Vâ5TD6ö×F–&–Æ—G’—2Væ&ÆVBàÐ¢%dÄÄÕô5TDô4ôÕD”$”Ä•E•õD‚#¢ÆÖ&F¢÷2æVçf—&öâævWB€Ð¢%dÄÄÕô5TDô4ôÕD”$”Ä•E•õD‚"ÂæöæPÐ¢’ÀÐ¢26¶—ÖöFVÂæÖRfÆ–FF–öâ–â÷Vä’’&WVW7G2àÐ¢2v†Vâ6WBFòÂç’ÖöFVÂæÖRv–ÆÂ&R66WFVB–âF†RvÖöFVÂrf–VÆ@Ð¢2öb’&WVW7G2âF†—2—2W6VgVÂf÷"&÷‡’övFWv’66Væ&–÷2v†W&PÐ¢2F†R7GVÂÖöFVÂ—26W'fVB'WBF–ffW&VçBæÖW2Ö’&RW6VB–â&WVW7G2àÐ¢%dÄÄÕõ4´•ôÔôDTÅôäÔUõdÄ”DD”ôâ#¢ÆÖ&F¢€Ð¢÷2ævWFVçb‚%dÄÄÕõ4´•ôÔôDTÅôäÔUõdÄ”DD”ôâ"Â#"’ç7G&—‚’æÆ÷vW"‚Ð¢–â‚#"Â'G'VR"Ð¢’ÀÐ¢2v†WF†W"—B—266ÆRWÆVæ6‚Væv–æRf÷"VÆ7F–2UÀÐ¢26†÷VÆBöæÇ’&R6WB'’Væv–æT6÷&T6Æ–VçBàÐ¢%dÄÄÕôTÄ5D”5ôUõ44ÄUõUôÄTä4‚#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôTÄ5D”5ôUõ44ÄUõUôÄTä4‚"Â#"’Ð¢’ÀÐ¢2v†WF†W"Fòv—Bf÷"ÆÂ&WVW7G2FòG&–â&Vf÷&R6VæF–ærF†PÐ¢266Æ–ær6öÖÖæB–âVÆ7F–2UàÐ¢%dÄÄÕôTÄ5D”5ôUôE$”åõ$UTU5E2#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôTÄ5D”5ôUôE$”åõ$UTU5E2"Â#"’Ð¢’ÀÐ¢2–b6WBFòÂVæ&ÆR5TDw&‚ÖVÖ÷'’W7F–ÖF–öâGW&–ærÖVÖ÷'’&öf–Æ–æràÐ¢2F†—2&öf–ÆW25TDw&‚ÖVÖ÷'’W6vRFò&÷f–FRÖ÷&R67W&FRµb66†PÐ¢2ÖVÖ÷'’ÆÆö6F–öââVæ&ÆVB'’FVfVÇB2öbcã#ã Ð¢%dÄÄÕôÔTÔõ%•õ$ôd”ÄU%ôU5D”ÔDUô5TDu$…2#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôÔTÔõ%•õ$ôd”ÄU%ôU5D”ÔDUô5TDu$…2"Â#"’Ð¢’ÀÐ¢2ä•„ÂUVçf—&öæÖVçBf&–&ÆW0Ð¢%dÄÄÕôä•„ÅôUôÔ…ôåTÕõ$äµ2#¢ÆÖ&F¢–çB€Ð¢÷2ævWFVçb‚%dÄÄÕôä•„ÅôUôÔ…ôåTÕõ$äµ2"Â#3""Ð¢’ÀÐ¢2v†WF†W"Væ&ÆR…Rw&‚öâ–çFVÂuPÐ¢%dÄÄÕõ…UôTä$ÄUõ…Uôu$‚#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõ…UôTä$ÄUõ…Uôu$‚"Â#"’Ð¢’ÀÐ¢2v†WF†W"W6R‡R7V6–f–26×ÆR¶W&æVÀÐ¢%dÄÄÕõ…UõU4Uõ4ÕÄU%ô´U$äTÂ#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõ…UõU4Uõ4ÕÄU%ô´U$äTÂ"Â#"’Ð¢’ÀÐ¢2Væ&ÆR6–×ÆRµböffÆöBàÐ¢%dÄÄÕõU4Uõ4”ÕÄUôµeôôddÄôB#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕõU4Uõ4”ÕÄUôµeôôddÄôB"Â#"’Ð¢’ÀÐ¢2v†WF†W"FòVæ&ÆRGVÂ7VF7G&V×2f÷"Æõ$6ö×WFF–öàÐ¢%dÄÄÕôÄõ$ôTä$ÄUôETÅõ5E$TÒ#¢ÆÖ&F¢&ööÂ€Ð¢–çB†÷2ævWFVçb‚%dÄÄÕôÄõ$ôTä$ÄUôETÅõ5E$TÒ"Â#"’Ð¢’ÀÐ§ÐÐ Ð Ð¢2ÒÓƒÂÒÒ¶VæC¦Vçb×f'2ÖFVf–æ—F–öåÐÐ Ð Ð¦FVbõövWFGG%õò†æÖS¢7G"“ Ð¢"" Ð¢vWG2Vçf—&öæÖVçBf&–&ÆW2Æ¦–Ç’àÐ Ð¢äõDS¢gFW"Væ&ÆUöVçg5ö66†R‚’–çfö6F–öâ‡v†–6‚G&–vvW&VBgFW"6W'f–6PÐ¢–æ—F–Æ—¦F–öâ’ÂÆÂVçf—&öæÖVçBf&–&ÆW2v–ÆÂ&R66†VBàÐ¢"" Ð¢–bæÖR–âVçf—&öæÖVçE÷f&–&ÆW3 Ð¢&WGW&âVçf—&öæÖVçE÷f&–&ÆW5¶æÖUÒ‚Ð¢&—6RGG&–'WFTW'&÷"†b&ÖöGVÆRµõöæÖUõò'Ò†2æòGG&–'WFR¶æÖR'Ò"Ð Ð Ð¦FVbö—5öVçg5ö66†UöVæ&ÆVB‚’Óâ&ööÃ Ð¢""$6†V6¶VB–bõövWFGG%õò—2w&VBv—F‚gVæ7FööÇ2æ66†R"" Ð¢vÆö&ÂõövWFGG%õðÐ¢&WGW&â†6GG"…õövWFGG%õòÂ&66†Uö6ÆV""Ð Ð Ð¦FVbVæ&ÆUöVçg5ö66†R‚’ÓâæöæS Ð¢"" Ð¢Væ&ÆW266†–æröbVçf—&öæÖVçBf&–&ÆW2âF†—2—2W6VgVÂf÷"W&f÷&Öæ6PÐ¢&V6öç2Â2—Bfö–G2F†RæVVBFò&RÖWfÇVFRVçf—&öæÖVçBf&–&ÆW2öàÐ¢WfW'’6ÆÂàÐ Ð¢äõDS¢7W'&VçFÇ’Â—Bw2–çfö¶VBgFW"6W'f–6R–æ—F–Æ—¦F–öâFò&VGV6PÐ¢'VçF–ÖR÷fW&†VBâF†—2Ç6òÖVç2F†BVçf—&öæÖVçBf&–&ÆW26†÷VÆBäõ@Ð¢&RWFFVBgFW"F†R6W'f–6R—2–æ—F–Æ—¦VBàÐ¢"" Ð¢–bö—5öVçg5ö66†UöVæ&ÆVB‚“ Ð¢2fö–Bw&–ærgVæ7FööÇ2æ66†R×VÇF—ÆRF–ÖW0Ð¢&WGW&àÐ¢2FrõövWFGG%õòv—F‚gVæ7FööÇ2æ66†PÐ¢vÆö&ÂõövWFGG%õðÐ¢õövWFGG%õòÒgVæ7FööÇ2æ66†R…õövWFGG%õòÐ Ð¢266†RÆÂVçf—&öæÖVçBf&–&ÆW0Ð¢f÷"¶W’–âVçf—&öæÖVçE÷f&–&ÆW3 Ð¢õövWFGG%õò†¶W’Ð Ð Ð¦FVbF—6&ÆUöVçg5ö66†R‚’ÓâæöæS Ð¢"" Ð¢&W6WG2F†RVçf—&öæÖVçBf&–&ÆW266†Râ—B6÷VÆB&RW6VBFò—6öÆFRVçf—&öæÖVçG0Ð¢&WGvVVâVæ—BFW7G2àÐ¢"" Ð¢vÆö&ÂõövWFGG%õðÐ¢2–bõövWFGG%õò—2w&VB'’gVæ7F–öç2æ66†RÂVçw&F†R66†–ærÆ–W"àÐ¢–bö—5öVçg5ö66†UöVæ&ÆVB‚“ Ð¢76W'B†6GG"…õövWFGG%õòÂ%õ÷w&VEõò"Ð¢õövWFGG%õòÒõövWFGG%õòåõ÷w&VEõðÐ Ð Ð¦FVbõöF—%õò‚“ Ð¢&WGW&âÆ—7B†Vçf—&öæÖVçE÷f&–&ÆW2æ¶W—2‚’Ð Ð Ð¦FVb—5÷6WB†æÖS¢7G"“ Ð¢""$6†V6²–bâVçf—&öæÖVçBf&–&ÆR—2W‡Æ–6—FÇ’6WBâ"" Ð¢–bæÖR–âVçf—&öæÖVçE÷f&–&ÆW3 Ð¢&WGW&âæÖR–â÷2æVçf—&öàÐ¢&—6RGG&–'WFTW'&÷"†b&ÖöGVÆRµõöæÖUõò'Ò†2æòGG&–'WFR¶æÖR'Ò"Ð Ð Ð¦FVbfÆ–FFUöVçf—&öâ††&Eöf–Ã¢&ööÂ’ÓâæöæS Ð¢f÷"Vçb–â÷2æVçf—&öã Ð¢–bVçbç7F'G7v—F‚‚%dÄÄÕò"’æBVçbæ÷B–âVçf—&öæÖVçE÷f&–&ÆW3 Ð¢–b†&Eöf–Ã Ð¢&—6RfÇVTW'&÷"†b%Væ¶æ÷vâdÄÄÒVçf—&öæÖVçBf&–&ÆRFWFV7FVC¢¶VçgÒ"Ð¢VÇ6S Ð¢ÆövvW"çv&æ–ær‚%Væ¶æ÷vâdÄÄÒVçf—&öæÖVçBf&–&ÆRFWFV7FVC¢W2"ÂVçbÐ Ð Ð¦FVb6ö×–ÆUöf7F÷'2‚’ÓâF–7E·7G"Âö&¦V7EÓ Ð¢""%&WGW&âVçbf'2W6VBf÷"F÷&6‚æ6ö×–ÆR66†R¶W—2àÐ Ð¢7F'Bv—F‚WfW'’¶æ÷vâdÄÄÒVçbf#²G&÷VçG&–W2–â–væ÷&VEöf7F÷'6°Ð¢†6‚WfW'—F†–ærVÇ6RâF†—2¶VW2F†R66†R¶W’Æ–væVB7&÷72v÷&¶W'2â"" Ð Ð¢–væ÷&VEöf7F÷'3¢6WE·7G%ÒÒ°Ð¢$Ô…ô¤ô%2"ÀÐ¢%dÄÄÕõ%5ô$4UõD‚"ÀÐ¢%dÄÄÕõU4UôÔôDTÅ44õR"ÀÐ¢%dÄÄÕõ$”ät%TddU%õt$ä”äuô”åDU%dÂ"ÀÐ¢%dÄÄÕôDT%TuôETÕõD‚"ÀÐ¢%dÄÄÕõõ%B"ÀÐ¢%dÄÄÕô44„Uõ$ôõB"ÀÐ¢$ÄEôÄ”%$%•õD‚"ÀÐ¢%dÄÄÕõ4U%dU%ôDUeôÔôDR"ÀÐ¢%dÄÄÕôEôÔ5DU%ô•"ÀÐ¢%dÄÄÕôEôÔ5DU%õõ%B"ÀÐ¢%dÄÄÕõ$äDôÔ•¤UôEôETÔÕ•ô”åUE2"ÀÐ¢%dÄÄÕô4•õU4Uõ32"ÀÐ¢%dÄÄÕôÔôDTÅõ$TD•$T5EõD‚"ÀÐ¢%dÄÄÕô„õ5Eô•"ÀÐ¢%dÄÄÕôdõ$4UôõEôÄôB"ÀÐ¢%35ô44U55ô´U•ô”B"ÀÐ¢%35õ4T5$UEô44U55ô´U’"ÀÐ¢%35ôTäEô”åEõU$Â"ÀÐ¢%dÄÄÕõU4tUõ5DE5õ4U%dU""ÀÐ¢%dÄÄÕôäõõU4tUõ5DE2"ÀÐ¢%dÄÄÕôDõôäõEõE$4²"ÀÐ¢%dÄÄÕôÄôtt”äuôÄUdTÂ"ÀÐ¢%dÄÄÕôÄôtt”äuõ$Td•‚"ÀÐ¢%dÄÄÕôÄôtt”äuõ5E$TÒ"ÀÐ¢%dÄÄÕôÄôtt”äuô4ôäd”uõD‚"ÀÐ¢%dÄÄÕôÄôtt”äuô4ôÄõ""ÀÐ¢%dÄÄÕôÄôuõ5DE5ô”åDU%dÂ"ÀÐ¢%dÄÄÕôDT%TuôÄôuô•õ4U%dU%õ$U5ôå4R"ÀÐ¢%dÄÄÕõETäTEô4ôäd”uôdôÄDU""ÀÐ¢%dÄÄÕôTät”äUô•DU$D”ôåõD”ÔTõUEõ2"ÀÐ¢%dÄÄÕô…EEõD”ÔTõUEô´TUôÄ•dR"ÀÐ¢%dÄÄÕôU„T5UDUôÔôDTÅõD”ÔTõUEõ4T4ôäE2"ÀÐ¢%dÄÄÕô´TUôÄ•dUôôåôTät”äUôDTD‚"ÀÐ¢%dÄÄÕô”ÔtUôdUD4…õD”ÔTõUB"ÀÐ¢%dÄÄÕõd”DTõôdUD4…õD”ÔTõUB"ÀÐ¢%dÄÄÕôTD”õôdUD4…õD”ÔTõUB"ÀÐ¢%dÄÄÕôÔTD”ô44„R"ÀÐ¢%dÄÄÕôÔTD”ô44„UôÔ…õ4•¤UôÔ""ÀÐ¢%dÄÄÕôÔTD”ô44„UõEDÅô„õU%2"ÀÐ¢%dÄÄÕôÔTD”ôdUD4…ôÔ…õ$UE$”U2"ÀÐ¢%dÄÄÕôÔTD”õU$ÅôÄÄõuõ$TD•$T5E2"ÀÐ¢%dÄÄÕôÔTD”ôÄôD”äuõD…$TEô4õTåB"ÀÐ¢%dÄÄÕôÔ…ôTD”õô4Ä•ôd”ÄU4•¤UôÔ""ÀÐ¢%dÄÄÕõd”DTõôÄôDU%ô$4´TäB"ÀÐ¢%dÄÄÕôÔTD”ô4ôääT5Dõ""ÀÐ¢%dÄÄÕôô$¤T5Eõ5Dõ$tUõ4„Õô%TddU%ôäÔR"ÀÐ¢%dÄÄÕô54UE5ô44„R"ÀÐ¢%dÄÄÕô54UE5ô44„UôÔôDTÅô4ÄTâ"ÀÐ¢%dÄÄÕõtõ$´U%ôÕTÅD•$ô5ôÔUD„ôB"ÀÐ¢%dÄÄÕôTä$ÄUõcôÕTÅD•$ô4U54”är"ÀÐ¢%dÄÄÕõcôõUEUEõ$ô5ô4…Täµõ4•¤R"ÀÐ¢%dÄÄÕô5Uôµd44„Uõ54R"ÀÐ¢%dÄÄÕô5UôÔôUõ$U4²"ÀÐ¢%dÄÄÕõ¤TåDõ$4…õtT”t…Eõ$U4²"ÀÐ¢%dÄÄÕõDU5Eôdõ$4UôÄôEôdõ$ÔB"ÀÐ¢%dÄÄÕôTä$ÄUô5TDô4ôÕD”$”Ä•E’"ÀÐ¢%dÄÄÕô5TDô4ôÕD”$”Ä•E•õD‚"ÀÐ¢%dÄÄÕõ4´•ôÔôDTÅôäÔUõdÄ”DD”ôâ"ÀÐ¢$Äô4Åõ$ä²"ÀÐ¢$5TDõd•4”$ÄUôDUd”4U2"ÀÐ¢$äõô4ôÄõ""ÀÐ¢ÐÐ Ð¢g&öÒfÆÆÒæ6öæf–rçWF–Ç2–×÷'Bæ÷&ÖÆ—¦U÷fÇVPÐ Ð¢f7F÷'3¢F–7E·7G"Âö&¦V7EÒÒ·ÐÐ¢f÷"f7F÷"ÂvWGFW"–âVçf—&öæÖVçE÷f&–&ÆW2æ—FV×2‚“ Ð¢–bf7F÷"–â–væ÷&VEöf7F÷'3 Ð¢6öçF–çVPÐ Ð¢G'“ Ð¢&rÒvWGFW"‚Ð¢W†6WBW†6WF–öâ2W†3¢2&vÖ¢æò6÷fW"ÒFVfVç6—fRÆövv–æpÐ¢ÆövvW"çv&æ–ær€Ð¢%6¶—–ærVçf—&öæÖVçBf&–&ÆRW2v†–ÆR†6†–ær6ö×–ÆRf7F÷'3¢W2"ÀÐ¢f7F÷"ÀÐ¢W†2ÀÐ¢Ð¢6öçF–çVPÐ Ð¢f7F÷'5¶f7F÷%ÒÒæ÷&ÖÆ—¦U÷fÇVR‡&rÐ Ð¢&•öæ÷6WEöVçe÷f'2Ò°Ð¢2&VfW"FðÐ¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2öçf–F–öwRç’4ÃÐ¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2öÖEöwRç’4ÃÐ¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö#“vC#F##363&&C†VCvF#sC–ƒ&SS“C##&#V2÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2öÖEöwRç’4Ã Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2öçRç’4Ã Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2ö‡Rç’4Ã Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2öæWW&öâç’4Ã@Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2÷GRç’4Ã3€Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2ö–çFVÅöwRç’4Ã Ð¢2‡GG3¢òöv—F‡V"æ6öÒ÷&’×&ö¦V7B÷&’ö&Æö"ö3SƒF#V“v#s“6CFVcsVcƒS3vCsVf&C"÷—F†öâ÷&’õ÷&—fFRö66VÆW&F÷'2÷&&Æâç’4Ã Ð¢%$•ôU…U$”ÔTåDÅôäõ4UEô5TDõd•4”$ÄUôDUd”4U2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEõ$ô5%õd•4”$ÄUôDUd”4U2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEô„•õd•4”$ÄUôDUd”4U2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEô44TäEõ%Eõd•4”$ÄUôDUd”4U2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEô„$äõd•4”$ÄUôÔôETÄU2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEôäUU$ôåõ%Eõd•4”$ÄUô4õ$U2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEõEUõd•4”$ÄUô4„•2"ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEôôäT•ôDUd”4Uõ4TÄT5Dõ""ÀÐ¢%$•ôU…U$”ÔTåDÅôäõ4UEõ$$Äåõ%Eõd•4”$ÄUôDUd”4U2"ÀÐ¢ÐÐ Ð¢f÷"f"–â&•öæ÷6WEöVçe÷f'3 Ð¢f7F÷'5·f%ÒÒæ÷&ÖÆ—¦U÷fÇVR†÷2ævWFVçb‡f"’Ð Ð¢&WGW&âf7F÷'0Ð