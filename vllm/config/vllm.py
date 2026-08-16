# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

import torch
from packaging.version import Version
from pydantic import ConfigDict, Field, model_validator

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.utils import random_uuid
from vllm.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .ec_transfer import ECTransferConfig
from .kernel import KernelConfig
from .kv_events import KVEventsConfig
from .kv_transfer import KVTransferConfig
from .load import LoadConfig
from .lora import LoRAConfig
from .mamba import MambaConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .offload import OffloadConfig
from .parallel import ParallelConfig
from .profiler import ProfilerConfig
from .reasoning import ReasoningConfig
from .scheduler import SchedulerConfig
from .speculative import EagleModelTypes, NgramGPUTypes, SpeculativeConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config, replace
from .weight_transfer import WeightTransferConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
else:
    PretrainedConfig = Any

    QuantizationConfig = Any

    KVCacheConfig = Any

logger = init_logger(__name__)


class OptimizationLevel(IntEnum):
    """Optimization level enum."""

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    O3 = 3
    """O3: Currently the same as -O2s."""


PerformanceMode = Literal["balanced", "interactivity", "throughput"]

IS_QUANTIZED = False
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# See https://github.com/vllm-project/vllm/issues/25689.


def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""

    return (
        cfg.compilation_config.is_custom_op_enabled("rms_norm")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or cfg.kernel_config.ir_op_priority.rms_norm[0] != "native"
    )


def enable_act_fusion(cfg: "VllmConfig") -> bool:
    """
    Enable if either SiLU+Mul or quant FP8 custom op is active;
    otherwise Inductor handles fusion.
    Also enable for FP4 models as FP4 quant is always custom so Inductor cannot fuse it.
    """
    return (
        cfg.compilation_config.is_custom_op_enabled("silu_and_mul")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
    )


def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
    """Enable if TP > 1 and Hopper/Blackwell and flashinfer installed."""
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer

    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        return (
            rocm_aiter_ops.is_enabled() and cfg.parallel_config.tensor_parallel_size > 1
        )

    return (
        cfg.parallel_config.tensor_parallel_size > 1
        and current_platform.is_cuda()
        and has_flashinfer()
        and (
            current_platform.is_device_capability_family(100)
            or current_platform.is_device_capability(90)
        )
    )


def enable_rope_kvcache_fusion(cfg: "VllmConfig") -> bool:
    """Enable if rotary embedding custom op is active and
    use_inductor_graph_partition is enabled.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    return (
        rocm_aiter_ops.is_enabled()
        and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
        and (
            cfg.compilation_config.use_inductor_graph_partition
            or not cfg.compilation_config.splitting_ops_contain_kv_cache_update()
        )
    )


def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
    """Enable if using AITER RMSNorm and hidden size is 2880 i.e. gpt-oss."""

    return (
        cfg.kernel_config.ir_op_priority.fused_add_rms_norm[0] == "aiter"
        and cfg.model_config is not None
        and cfg.model_config.get_hidden_size() == 2880
    )


def enable_mla_dual_rms_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable MLA dual RMS norm fusion when AITer has fused_qk_rmsnorm."""
    from vllm._aiter_ops import check_aiter_fused_qk_rmsnorm, rocm_aiter_ops

    return rocm_aiter_ops.is_enabled() and check_aiter_fused_qk_rmsnorm()


OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
            "fuse_mla_dual_rms_norm": False,
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        # Disabled for now due to correctness issues:
        # https://github.com/flashinfer-ai/flashinfer/issues/3197
        "enable_flashinfer_autotune": False,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        # Disabled for now due to correctness issues:
        # https://github.com/flashinfer-ai/flashinfer/issues/3197
        "enable_flashinfer_autotune": False,
    },
}
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_mla_dual_rms_norm": enable_mla_dual_rms_norm_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = None  # type: ignore[assignment]
    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig.default_factory,
    )
    """Scheduler configuration."""
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    """Load configuration."""
    offload_config: OffloadConfig = Field(default_factory=OffloadConfig)
    """Model weight offloading configuration."""
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    """Attention configuration."""
    mamba_config: MambaConfig = Field(default_factory=MambaConfig)
    """Mamba configuration."""
    kernel_config: KernelConfig = Field(default_factory=KernelConfig)
    """Kernel configuration."""
    lora_config: LoRAConfig | None = None
    """LoRA configuration."""
    speculative_config: SpeculativeConfig | None = None
    """Speculative decoding configuration."""
    structured_outputs_config: StructuredOutputsConfig = Field(
        default_factory=StructuredOutputsConfig
    )
    """Structured outputs configuration."""
    observability_config: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    """Observability configuration."""
    quant_config: QuantizationConfig | None = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, one can append compilation arguments via
    -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

    You can specify the full compilation config like so:
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig)
    """Profiling configuration."""
    kv_transfer_config: KVTransferConfig | None = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: KVEventsConfig | None = None
    """The configurations for event publishing."""
    ec_transfer_config: ECTransferConfig | None = None
    """The configurations for distributed EC cache transfer."""
    reasoning_config: ReasoningConfig | None = None
    """The configurations for reasoning model."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the vLLM instance."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -O2 is used by default. See OptimizationLevel for full
    description."""

    performance_mode: PerformanceMode = "balanced"
    """Performance mode for runtime behavior, 'balanced' is the default.
    'interactivity' favors low end-to-end per-request latency at small batch
    sizes (fine-grained CUDA graphs, latency-oriented kernels).
    'throughput' favors aggregate tokens/sec at high concurrency (larger CUDA
    graphs, more aggressive batching, throughput-oriented kernels)."""

    weight_transfer_config: WeightTransferConfig | None = None
    """The configurations for weight transfer during RL training."""

    shutdown_timeout: int = Field(default=0, ge=0)
    """Shutdown grace period for in-flight requests. Shutdown will be delayed for
    up to this amount of time to allow already-running requests to complete. Any
    remaining requests are aborted once the timeout is reached.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        from vllm import __version__

        vllm_factors.append(__version__)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
            if (
                self.compilation_config
                and getattr(self.compilation_config, "compile_mm_encoder", False)
                and self.model_config.multimodal_config
            ):
                vllm_facÛmùòÚ$z{-®éÜj×6öæf–ræÆöEöf÷&ÖBæ÷B–â€Ð¢''Væ•÷7G&VÖW""ÀÐ¢''Væ•÷7G&VÖW%÷6†&FVB"ÀÐ¢“ Ð¢&—6RfÇVTW'&÷"€Ð¢b%FòÆöBÖöFVÂg&öÒö&¦V7B7F÷&vR…32ôt52ô§W&R’Â Ð¢b"vÆöEöf÷&ÖBr×W7B&Rw'Væ•÷7G&VÖW"r÷" Ð¢b"w'Væ•÷7G&VÖW%÷6†&FVBrÂ Ð¢b&'WBv÷Bw·6VÆbæÆöEö6öæf–ræÆöEöf÷&ÖGÒrâ Ð¢b$ÖöFVÃ¢·6VÆbæÖöFVÅö6öæf–ræÖöFVÇÒ Ð¢Ð Ð¢FVb6ö×–ÆUöFV'VuöGV×÷F‚‡6VÆb’ÓâF‚ÂæöæS Ð¢""%&WGW&ç2&æ²Öv&RF‚f÷"GV×–æpÐ¢F÷&6‚æ6ö×–ÆRFV'Vr–æf÷&ÖF–öâàÐ¢"" Ð¢–b6VÆbæ6ö×–ÆF–öåö6öæf–ræFV'VuöGV×÷F‚—2æöæS Ð¢&WGW&âæöæPÐ¢G÷&æ²Ò6VÆbç&ÆÆVÅö6öæf–rç&æ°Ð¢G÷&æ²Ò6VÆbç&ÆÆVÅö6öæf–ræFF÷&ÆÆVÅö–æFW€Ð¢VæE÷F‚Òb'&æµ÷·G÷&æ·ÕöG÷¶G÷&æ·Ò Ð¢F‚Ò6VÆbæ6ö×–ÆF–öåö6öæf–ræFV'VuöGV×÷F‚òVæE÷F€Ð¢&WGW&âF€Ð Ð¢FVbõ÷7G%õò‡6VÆb“ Ð¢&WGW&â€Ð¢b&ÖöFVÃ×·6VÆbæÖöFVÅö6öæf–ræÖöFVÂ'ÒÂ Ð¢b'7V7VÆF—fUö6öæf–s×·6VÆbç7V7VÆF—fUö6öæf–r'ÒÂ Ð¢b'Fö¶Væ—¦W#×·6VÆbæÖöFVÅö6öæf–rçFö¶Væ—¦W"'ÒÂ Ð¢b'6¶—÷Fö¶Væ—¦W%ö–æ—C×·6VÆbæÖöFVÅö6öæf–rç6¶—÷Fö¶Væ—¦W%ö–æ—GÒÂ Ð¢b'Fö¶Væ—¦W%öÖöFS×·6VÆbæÖöFVÅö6öæf–rçFö¶Væ—¦W%öÖöFWÒÂ Ð¢b'&Wf—6–öã×·6VÆbæÖöFVÅö6öæf–rç&Wf—6–öçÒÂ Ð¢b'Fö¶Væ—¦W%÷&Wf—6–öã×·6VÆbæÖöFVÅö6öæf–rçFö¶Væ—¦W%÷&Wf—6–öçÒÂ Ð¢b'G'W7E÷&VÖ÷FUö6öFS×·6VÆbæÖöFVÅö6öæf–rçG'W7E÷&VÖ÷FUö6öFWÒÂ Ð¢b&GG—S×·6VÆbæÖöFVÅö6öæf–ræGG—WÒÂ Ð¢b&Ö…÷6WöÆVã×·6VÆbæÖöFVÅö6öæf–ræÖ…öÖöFVÅöÆVçÒÂ Ð¢b&F÷væÆöEöF—#×·6VÆbæÆöEö6öæf–ræF÷væÆöEöF—"'ÒÂ Ð¢b&ÆöEöf÷&ÖC×·6VÆbæÆöEö6öæf–ræÆöEöf÷&ÖGÒÂ Ð¢b'FVç6÷%÷&ÆÆVÅ÷6—¦S×·6VÆbç&ÆÆVÅö6öæf–rçFVç6÷%÷&ÆÆVÅ÷6—¦WÒÂ"2æ÷Ð¢b'—VÆ–æU÷&ÆÆVÅ÷6—¦S×·6VÆbç&ÆÆVÅö6öæf–rç—VÆ–æU÷&ÆÆVÅ÷6—¦WÒÂ"2æ÷Ð¢b&FF÷&ÆÆVÅ÷6—¦S×·6VÆbç&ÆÆVÅö6öæf–ræFF÷&ÆÆVÅ÷6—¦WÒÂ"2æ÷Ð¢b&FV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦S×·6VÆbç&ÆÆVÅö6öæf–ræFV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦WÒÂ"2æ÷Ð¢b&F7ö6öÖÕö&6¶VæC×·6VÆbç&ÆÆVÅö6öæf–ræF7ö6öÖÕö&6¶VæGÒÂ"2æ÷Ð¢b&F—6&ÆUö7W7FöÕöÆÅ÷&VGV6S×·6VÆbç&ÆÆVÅö6öæf–ræF—6&ÆUö7W7FöÕöÆÅ÷&VGV6WÒÂ"2æ÷Ð¢b'VçF—¦F–öã×·6VÆbæÖöFVÅö6öæf–rçVçF—¦F–öçÒÂ Ð¢b'VçF—¦F–öåö6öæf–s×·6VÆbæÖöFVÅö6öæf–rçVçF—¦F–öåö6öæf–wÒÂ"2æ÷Ð¢b&Væf÷&6UöVvW#×·6VÆbæÖöFVÅö6öæf–ræVæf÷&6UöVvW'ÒÂ Ð¢b&Væ&ÆU÷&WGW&å÷&÷WFVEöW‡W'G3×·6VÆbæÖöFVÅö6öæf–ræVæ&ÆU÷&WGW&å÷&÷WFVEöW‡W'G7ÒÂ"2æ÷Ð¢b&·eö66†UöGG—S×·6VÆbæ66†Uö6öæf–ræ66†UöGG—WÒÂ Ð¢b&FWf–6Uö6öæf–s×·6VÆbæFWf–6Uö6öæf–ræFWf–6WÒÂ Ð¢b'7G'V7GW&VEö÷WGWG5ö6öæf–s×·6VÆbç7G'V7GW&VEö÷WGWG5ö6öæf–r'ÒÂ Ð¢b&ö'6W'f&–Æ—G•ö6öæf–s×·6VÆbæö'6W'f&–Æ—G•ö6öæf–r'ÒÂ Ð¢b'6VVC×·6VÆbæÖöFVÅö6öæf–rç6VVGÒÂ Ð¢b'6W'fVEöÖöFVÅöæÖS×·6VÆbæÖöFVÅö6öæf–rç6W'fVEöÖöFVÅöæÖWÒÂ Ð¢b&Væ&ÆU÷&Vf—…ö66†–æs×·6VÆbæ66†Uö6öæf–ræVæ&ÆU÷&Vf—…ö66†–æwÒÂ Ð¢b&Væ&ÆUö6‡Væ¶VE÷&Vf–ÆÃ×·6VÆbç66†VGVÆW%ö6öæf–ræVæ&ÆUö6‡Væ¶VE÷&Vf–ÆÇÒÂ"2æ÷Ð¢b'ööÆW%ö6öæf–s×·6VÆbæÖöFVÅö6öæf–rçööÆW%ö6öæf–r'ÒÂ Ð¢b&6ö×–ÆF–öåö6öæf–s×·6VÆbæ6ö×–ÆF–öåö6öæf–r'ÒÂ Ð¢b&¶W&æVÅö6öæf–s×·6VÆbæ¶W&æVÅö6öæf–r'Ò Ð¢Ð Ð¢FVb÷fÆ–FFU÷c%öÖöFVÅ÷'VææW"‡6VÆb’ÓâæöæS Ð¢""$6†V6²f÷"fVGW&W2æ÷B–WB7W÷'FVB'’F†Rc"ÖöFVÂ'VææW"â"" Ð¢Vç7W÷'FVC¢Æ—7E·7G%ÒÒµÐÐ Ð¢–b6VÆbæÖöFVÅö6öæf–r—2æ÷BæöæRæB6VÆbæÖöFVÅö6öæf–ræ†5ö–ææW%÷7FFS Ð¢Vç7W÷'FVBæVæB‚&‡–'&–BöÖÖ&ÖöFVÇ2"Ð Ð¢–b6VÆbç&ÆÆVÅö6öæf–rç&Vf–ÆÅö6öçFW‡E÷&ÆÆVÅ÷6—¦Râ Ð¢Vç7W÷'FVBæVæB‚'&Vf–ÆÂ6öçFW‡B&ÆÆVÆ—6Ò"Ð Ð¢–b€Ð¢6VÆbç7V7VÆF—fUö6öæf–r—2æ÷BæöæPÐ¢æB6VÆbç7V7VÆF—fUö6öæf–ræÖWF†öBæ÷B–â‚&VvÆR"Â&VvÆS2"Â&×G"Ð¢“ Ð¢Vç7W÷'FVBæVæB†b'7V7VÆF—fRÖWF†öBw·6VÆbç7V7VÆF—fUö6öæf–ræÖWF†öGÒr"Ð Ð¢–b6VÆbç&ÆÆVÅö6öæf–ræVæ&ÆUöF&ó Ð¢Vç7W÷'FVBæVæB‚&GVÂ&F6‚÷fW&Æ"Ð Ð¢–b€Ð¢6VÆbæÖöFVÅö6öæf–r—2æ÷BæöæPÐ¢æB6VÆbæÖöFVÅö6öæf–ræVæ&ÆU÷&WGW&å÷&÷WFVEöW‡W'G0Ð¢“ Ð¢2v–ÆÂ&RFFVB'’‡GG3¢òöv—F‡V"æ6öÒ÷fÆÆÒ×&ö¦V7B÷fÆÆÒ÷VÆÂó3ƒc0Ð¢Vç7W÷'FVBæVæB‚'&÷WFVBW‡W'G26GW&R"Ð Ð¢–b6VÆbæÖöFVÅö6öæf–r—2æ÷BæöæRæB6VÆbæÖöFVÅö6öæf–ræÆöv—G5÷&ö6W76÷'3 Ð¢Vç7W÷'FVBæVæB‚&7W7FöÒÆöv—G2&ö6W76÷'2"Ð Ð¢–b6VÆbæ66†Uö6öæf–ræ·e÷6†&–æuöf7E÷&Vf–ÆÃ Ð¢2v–ÆÂ&RFFVB'’‡GG3¢òöv—F‡V"æ6öÒ÷fÆÆÒ×&ö¦V7B÷fÆÆÒ÷VÆÂó3SCPÐ¢Vç7W÷'FVBæVæB‚$µb6†&–ærf7B&Vf–ÆÂ"Ð Ð¢–b6VÆbæV5÷G&ç6fW%ö6öæf–r—2æ÷BæöæS Ð¢2v–ÆÂ&RFFVB'’‡GG3¢òöv—F‡V"æ6öÒ÷fÆÆÒ×&ö¦V7B÷fÆÆÒ÷VÆÂó3ƒ3“ Ð¢Vç7W÷'FVBæVæB‚$T2G&ç6fW""Ð Ð¢–bVç7W÷'FVC Ð¢&—6RfÇVTW'&÷"€Ð¢%dÄÄÕõU4Uõc%ôÔôDTÅõ%TääU"FöW2æ÷B–WB7W÷'C¢ Ð¢²"Â"æ¦ö–â‡Vç7W÷'FVBÐ¢Ð Ð¢FVb÷fÆ–FFU÷&WGW&å÷&÷WFVEöW‡W'G2‡6VÆb’ÓâæöæS Ð¢""%&V¦V7B&ÆÆVÆ—6Ò6öæf–wW&F–öç2æ÷B–WBfÆ–FFVBv—F€Ð¢ÒÖVæ&ÆR×&WGW&â×&÷WFVBÖW‡W'G2àÐ Ð¢fÆ–FFVB66÷R…"33““r“¢EÂUÂEÂ6–ævÆRÖæöFRæB×VÇF’ÖæöFRÀÐ¢&Vf—‚66†–ærÂæB7V7VÆF—fRFV6öF–ær„ÕEfÆ–FFVBVæB×FòÖVæC°Ð¢VvÆRôVvÆS2ôæw&ÒôÖVGW67W÷'FVB'’6öç7G'V7F–öâ6–æ6RF†PÐ¢&÷WF–ær'VffW"—2&÷VæBöæÇ’FòF†RF&vWBÖöFVÂæBfW&–f–VB×Fö¶VàÐ¢&÷WF–ærÆæG2BF†R6÷'&V7B÷6—F–öç2GW&–ærF†RÖ–âf÷'v&B’àÐ Ð¢÷WBÖöb×66÷R†&Æö6²VçF–ÂfÆ–FFVB“¢âÂ&Vf–ÆÂ6öçFW‡@Ð¢&ÆÆVÆ—6Ò…5’âÂFV6öFR6öçFW‡B&ÆÆVÆ—6Ò„D5’âÀÐ¢7–æ266†VGVÆ–æràÐ¢"" Ð¢Vç7W÷'FVC¢Æ—7E·7G%ÒÒµÐÐ Ð¢–b6VÆbç&ÆÆVÅö6öæf–rç—VÆ–æU÷&ÆÆVÅ÷6—¦Râ Ð¢Vç7W÷'FVBæVæB€Ð¢'—VÆ–æR&ÆÆVÆ—6Ò Ð¢b"‡—VÆ–æU÷&ÆÆVÅ÷6—¦SÒ Ð¢b'·6VÆbç&ÆÆVÅö6öæf–rç—VÆ–æU÷&ÆÆVÅ÷6—¦WÒ’ Ð¢Ð¢–b6VÆbç&ÆÆVÅö6öæf–rç&Vf–ÆÅö6öçFW‡E÷&ÆÆVÅ÷6—¦Râ Ð¢Vç7W÷'FVBæVæB€Ð¢'&Vf–ÆÂ6öçFW‡B&ÆÆVÆ—6Ò Ð¢b"‡&Vf–ÆÅö6öçFW‡E÷&ÆÆVÅ÷6—¦SÒ Ð¢b'·6VÆbç&ÆÆVÅö6öæf–rç&Vf–ÆÅö6öçFW‡E÷&ÆÆVÅ÷6—¦WÒ’ Ð¢Ð¢–b6VÆbç&ÆÆVÅö6öæf–ræFV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦Râ Ð¢Vç7W÷'FVBæVæB€Ð¢&FV6öFR6öçFW‡B&ÆÆVÆ—6Ò Ð¢b"†FV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦SÒ Ð¢b'·6VÆbç&ÆÆVÅö6öæf–ræFV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦WÒ’ Ð¢Ð¢–b6VÆbç66†VGVÆW%ö6öæf–ræ7–æ5÷66†VGVÆ–æs Ð¢Vç7W÷'FVBæVæB‚&7–æ266†VGVÆ–ær"Ð Ð¢–bVç7W÷'FVC Ð¢&—6RfÇVTW'&÷"€Ð¢"ÒÖVæ&ÆR×&WGW&â×&÷WFVBÖW‡W'G2—2æ÷B–WBfÆ–FFVBv—Fƒ¢ Ð¢²"Â"æ¦ö–â‡Vç7W÷'FVBÐ¢²"âF—6&ÆRF†W6RfVGW&W2÷"öÖ—B Ð¢"ÒÖVæ&ÆR×&WGW&â×&÷WFVBÖW‡W'G2â Ð¢Ð Ð¢FVbfÆ–FFUö&Æö6µ÷6—¦R‡6VÆb’ÓâæöæS Ð¢""%fÆ–FFR&Æö6µ÷6—¦Rv–ç7BD5æBÖÖ&6öç7G&–çG2àÐ Ð¢6ÆÆVBgFW"ÆFf÷&ÒçWFFUö&Æö6µ÷6—¦Uöf÷%ö&6¶VæB‚’†0Ð¢f–æÆ—6VB&Æö6µ÷6—¦RàÐ¢"" Ð¢&Æö6µ÷6—¦RÒ6VÆbæ66†Uö6öæf–ræ&Æö6µ÷6—¦PÐ Ð¢2D5–çFW&ÆVfR×6—¦R6ö×F–&–Æ—GÐ¢–b6VÆbç&ÆÆVÅö6öæf–ræFV6öFUö6öçFW‡E÷&ÆÆVÅ÷6—¦Râ Ð¢–b6VÆbç&ÆÆVÅö6öæf–ræF7ö·eö66†Uö–çFW&ÆVfU÷6—¦RâæB€Ð¢6VÆbç&ÆÆVÅö6öæf–ræ7ö·eö66†Uö–çFW&ÆVfU÷6—¦PÐ¢Ò6VÆbç&ÆÆVÅö6öæf–ræF7ö·eö66†Uö–çFW&ÆVfU÷6—¦PÐ¢“ Ð¢6VÆbç&ÆÆVÅö6öæf–ræ7ö·eö66†Uö–çFW&ÆVfU÷6—¦RÒ€Ð¢6VÆbç&ÆÆVÅö6öæf–ræF7ö·eö66†Uö–çFW&ÆVfU÷6—¦PÐ¢Ð¢ÆövvW"çv&æ–æuööæ6R€Ð¢&7ö·eö66†Uö–çFW&ÆVfU÷6—¦R—2÷fW'&–FFVâ'’F7ö·eö66†R Ð¢%ö–çFW&ÆVfU÷6—¦RâæBF7Ö·bÖ66†RÖ–çFW&ÆVfR×6—¦Rv–ÆÂ&R Ð¢&FW&V6FVBv†Vâ5—2gVÆÇ’7W÷'FVBâ Ð¢Ð¢76W'B€Ð¢6VÆbç&ÆÆVÅö6öæf–ræ7ö·eö66†Uö–çFW&ÆVfU÷6—¦RÃÒ&Æö6µ÷6—¦PÐ¢æB&Æö6µ÷6—¦RR6VÆbç&ÆÆVÅö6öæf–ræ7ö·eö66†Uö–çFW&ÆVfU÷6—¦RÓÒ Ð¢’Â€Ð¢b$&Æö6µ÷6—¦R‡¶&Æö6µ÷6—¦WÒ’6†÷VÆB&Rw&VFW" Ð¢'F†â÷"WVÂFòæBF—f—6–&ÆR'’7ö·eö66†Uö–çFW&ÆVfU÷6—¦R Ð¢b"‡·6VÆbç&ÆÆVÅö6öæf–ræ7ö·eö66†Uö–çFW&ÆVfU÷6—¦WÒ’â Ð¢Ð Ð¢2ÖÖ&66†RÆ–vâÖÖöFR6öç7G&–çG0Ð¢–b6VÆbæ66†Uö6öæf–ræÖÖ&ö66†UöÖöFRÓÒ&Æ–vâ# Ð¢76W'B&Æö6µ÷6—¦RÃÒ6VÆbç66†VGVÆW%ö6öæf–ræÖ…öçVÕö&F6†VE÷Fö¶Vç2Â€Ð¢$–âÖÖ&66†RÆ–vâÖöFRÂ&Æö6µ÷6—¦R Ð¢b"‡¶&Æö6µ÷6—¦WÒ’×W7B&RÃÒ Ð¢&Ö…öçVÕö&F6†VE÷Fö¶Vç2 Ð¢b"‡·6VÆbç66†VGVÆW%ö6öæf–ræÖ…öçVÕö&F6†VE÷Fö¶Vç7Ò’â Ð¢Ð¢–b6VÆbç66†VGVÆW%ö6öæf–ræÆöæu÷&Vf–ÆÅ÷Fö¶Vå÷F‡&W6†öÆBâ Ð¢76W'B6VÆbç66†VGVÆW%ö6öæf–ræÆöæu÷&Vf–ÆÅ÷Fö¶Vå÷F‡&W6†öÆBãÒ&Æö6µ÷6—¦PÐ¢76W'Bæ÷B6VÆbç66†VGVÆW%ö6öæf–ræF—6&ÆUö6‡Væ¶VEöÖÕö–çWBÂ€Ð¢$6‡Væ¶VBÔÒ–çWB—2&WV—&VB&V6W6RvRæVVBF†RfÆW†–&–Æ—G’ Ð¢'Fò66†VGVÆR×VÇF—ÆRöb&Æö6µ÷6—¦RFö¶Vç2WfVâ–bF†W’&R Ð¢&–âF†RÖ–FFÆRöbÖÒ–çWB Ð¢Ð¢2DôDó¢7W÷'BÆ–vâÖÖ&66†RÖöFRf÷"ÖöFVÂ'VææW"c Ð¢76W'Bæ÷BVçg2ådÄÄÕõU4Uõc%ôÔôDTÅõ%TääU"Â€Ð¢$ÖöFVÂ'VææW"c"†2æ÷B–WB7W÷'FVBÖÖ&ö66†UöÖöFSÒvÆ–vârâ Ð¢Ð Ð¢ÖöFVÅ÷fÆ–FF÷"†ÖöFSÒ&gFW""Ð¢FVbfÆ–FFUöçfgEö·eö66†U÷v—F…öÖÆ‡6VÆb’Óâ%fÆÆÔ6öæf–r# Ð¢–b6VÆbæÖöFVÅö6öæf–r—2æöæS Ð¢&WGW&â6VÆ`Ð¢–b6VÆbæ66†Uö6öæf–ræ66†UöGG—RÓÒ&çfgB"æB6VÆbæÖöFVÅö6öæf–rçW6UöÖÆ Ð¢&—6RfÇVTW'&÷"€Ð¢&çfgBµb66†R—2æ÷B7W÷'FVBv—F‚ÔÄ„×VÇF’Ö†VBÆFVçB Ð¢$GFVçF–öâ’&6¶VæG2âÆV6RW6RF–ffW&VçBÒÖ·bÖ66†RÖGG—R Ð¢"†RærâÂvg‚r÷"vWFòr’f÷"ÔÄÖöFVÇ27V6‚2FVW6VV²â Ð¢Ð¢&WGW&â6VÆ`Ð Ð¢ÖöFVÅ÷fÆ–FF÷"†ÖöFSÒ&gFW""Ð¢FVbfÆ–FFUöÖÖ&ö&Æö6µ÷6—¦R‡6VÆb’Óâ%fÆÆÔ6öæf–r# Ð¢–b6VÆbæÖöFVÅö6öæf–r—2æöæS Ð¢&WGW&â6VÆ`Ð¢ÖÖ&ö&Æö6µ÷6—¦Uö—5÷6WBÒ€Ð¢6VÆbæ66†Uö6öæf–ræÖÖ&ö&Æö6µ÷6—¦R—2æ÷BæöæPÐ¢æB6VÆbæ66†Uö6öæf–ræÖÖ&ö&Æö6µ÷6—¦RÒ6VÆbæÖöFVÅö6öæf–ræÖ…öÖöFVÅöÆVàÐ¢Ð¢–bÖÖ&ö&Æö6µ÷6—¦Uö—5÷6WBæBæ÷B6VÆbæ66†Uö6öæf–ræVæ&ÆU÷&Vf—…ö66†–æs Ð¢&—6RfÇVTW'&÷"€Ð¢"ÒÖÖÖ&Ö&Æö6²×6—¦R6âöæÇ’&R6WBv—F‚ÒÖVæ&ÆR×&Vf—‚Ö66†–ær Ð¢Ð¢&WGW&â6VÆ`Ð Ð Ð¥ö7W'&VçE÷fÆÆÕö6öæf–s¢fÆÆÔ6öæf–rÂæöæRÒæöæPÐ¥ö7W'&VçE÷&Vf—ƒ¢7G"ÂæöæRÒæöæPÐ Ð Ð¤6öçFW‡FÖævW Ð¦FVb6WEö7W'&VçE÷fÆÆÕö6öæf–r€Ð¢fÆÆÕö6öæf–s¢fÆÆÔ6öæf–rÂ6†V6µö6ö×–ÆSÔfÇ6RÂ&Vf—ƒ¢7G"ÂæöæRÒæöæPÐ¢“ Ð¢"" Ð¢FV×÷&&–Ç’6WBF†R7W'&VçBdÄÄÒ6öæf–ràÐ¢W6VBGW&–ærÖöFVÂ–æ—F–Æ—¦F–öâàÐ¢vR6fRF†R7W'&VçBdÄÄÒ6öæf–r–âvÆö&Âf&–&ÆRÀÐ¢6òF†BÆÂÖöGVÆW26â66W72—BÂRærâ7W7FöÒ÷0Ð¢6â66W72F†RdÄÄÒ6öæf–rFòFWFW&Ö–æR†÷rFòF—7F6‚àÐ¢"" Ð¢vÆö&Âö7W'&VçE÷fÆÆÕö6öæf–rÂö7W'&VçE÷&Vf—€Ð¢öÆE÷fÆÆÕö6öæf–rÒö7W'&VçE÷fÆÆÕö6öæf–pÐ¢öÆE÷&Vf—‚Òö7W'&VçE÷&Vf—€Ð¢g&öÒfÆÆÒæ6ö×–ÆF–öâæ6÷VçFW"–×÷'B6ö×–ÆF–öåö6÷VçFW Ð Ð¢çVÕöÖöFVÇ5÷6VVâÒ6ö×–ÆF–öåö6÷VçFW"æçVÕöÖöFVÇ5÷6VVàÐ¢G'“ Ð¢26ÆV"F†R6ö×–ÆF–öâ6öæf–r66†Rv†Vâ6öçFW‡B6†ævW2àÐ¢2F†—2—2æVVFVB6–æ6RF†RöÆB6öæf–rÖ’†fR&VVâ66W76V@Ð¢2æB66†VB&Vf÷&RF†RæWr6öæf–r—26WBàÐ¢vWEö66†VEö6ö×–ÆF–öåö6öæf–ræ66†Uö6ÆV"‚Ð Ð¢ö7W'&VçE÷fÆÆÕö6öæf–rÒfÆÆÕö6öæf–pÐ¢ö7W'&VçE÷&Vf—‚Ò&Vf—€Ð¢––VÆ@Ð¢W†6WBW†6WF–öã Ð¢&—6PÐ¢VÇ6S Ð¢–b6†V6µö6ö×–ÆS Ð¢fÆÆÕö6öæf–ræ6ö×–ÆF–öåö6öæf–ræ7W7FöÕö÷öÆöuö6†V6²‚Ð Ð¢–b€Ð¢6†V6µö6ö×–ÆPÐ¢æBfÆÆÕö6öæf–ræ6ö×–ÆF–öåö6öæf–ræÖöFRÓÒ6ö×–ÆF–öäÖöFRådÄÄÕô4ôÕ”ÄPÐ¢æB6ö×–ÆF–öåö6÷VçFW"æçVÕöÖöFVÇ5÷6VVâÓÒçVÕöÖöFVÇ5÷6VVàÐ¢“ Ð¢2–bF†RÖöFVÂ7W÷'G26ö×–ÆF–öâÀÐ¢26ö×–ÆF–öåö6÷VçFW"æçVÕöÖöFVÇ5÷6VVâ6†÷VÆB&R–æ7&V6V@Ð¢2'’BÆV7BàÐ¢2–b—B—2æ÷B–æ7&V6VBÂ—BÖVç2F†RÖöFVÂFöW2æ÷B7W÷'@Ð¢26ö×–ÆF–öâ†FöW2æ÷B†fR7W÷'E÷F÷&6…ö6ö×–ÆRFV6÷&F÷"’àÐ¢ÆövvW"çv&æ–ær€Ð¢&F÷&6‚æ6ö×–ÆV—2GW&æVBöâÂ'WBF†RÖöFVÂW2 Ð¢"FöW2æ÷B7W÷'B—BâÆV6R÷Vââ—77VRöâv—D‡V" Ð¢"–b–÷RvçB—BFò&R7W÷'FVBâ"ÀÐ¢fÆÆÕö6öæf–ræÖöFVÅö6öæf–ræÖöFVÂÀÐ¢Ð¢f–æÆÇ“ Ð¢ö7W'&VçE÷fÆÆÕö6öæf–rÒöÆE÷fÆÆÕö6öæf–pÐ¢ö7W'&VçE÷&Vf—‚ÒöÆE÷&Vf—€Ð¢26ÆV"F†R6ö×–ÆF–öâ6öæf–r66†Rv†Vâ6öçFW‡B6†ævW0Ð¢vWEö66†VEö6ö×–ÆF–öåö6öæf–ræ66†Uö6ÆV"‚Ð Ð Ð¤Ç'Uö66†R†Ö‡6—¦SÓÐ¦FVbvWEö66†VEö6ö×–ÆF–öåö6öæf–r‚“ Ð¢""$66†R6öæf–rFòfö–B&WVFVB6ÆÇ2FòvWEö7W'&VçE÷fÆÆÕö6öæf–r‚’"" Ð¢&WGW&âvWEö7W'&VçE÷fÆÆÕö6öæf–r‚’æ6ö×–ÆF–öåö6öæf–pÐ Ð Ð¦FVbvWEö7W'&VçE÷fÆÆÕö6öæf–r‚’ÓâfÆÆÔ6öæf–s Ð¢–bö7W'&VçE÷fÆÆÕö6öæf–r—2æöæS Ð¢&—6R76W'F–öäW'&÷"€Ð¢$7W'&VçBdÄÄÒ6öæf–r—2æ÷B6WBâF†—2G—–6ÆÇ’ÖVç2 Ð¢&vWEö7W'&VçE÷fÆÆÕö6öæf–r‚’v26ÆÆVB÷WG6–FRöb Ð¢'6WEö7W'&VçE÷fÆÆÕö6öæf–r‚’6öçFW‡BÂ÷"7W7FöÔ÷v2–ç7FçF–FVB Ð¢&BÖöGVÆR–×÷'BF–ÖR÷"ÖöFVÂf÷'v&BF–ÖRv†Vâ6öæf–r—2æ÷B6WBâ Ð¢$f÷"FW7G2F†BF—&V7FÇ’FW7B7W7FöÒ÷2öÖöGVÆW2ÂW6RF†R Ð¢"vFVfVÇE÷fÆÆÕö6öæf–rr—FW7Bf—‡GW&Rg&öÒFW7G2ö6öægFW7Bç’â Ð¢Ð¢&WGW&âö7W'&VçE÷fÆÆÕö6öæf–pÐ Ð Ð¦FVbvWEö7W'&VçE÷fÆÆÕö6öæf–uö÷%öæöæR‚’ÓâfÆÆÔ6öæf–rÂæöæS Ð¢&WGW&âö7W'&VçE÷fÆÆÕö6öæf–pÐ Ð Ð¥BÒG—Uf"‚%B"Ð Ð Ð¦FVbvWEöÆ–W'5ög&öÕ÷fÆÆÕö6öæf–r€Ð¢fÆÆÕö6öæf–s¢fÆÆÔ6öæf–rÀÐ¢Æ–W%÷G—S¢G—UµEÒÀÐ¢Æ–W%öæÖW3¢Æ—7E·7G%ÒÂæöæRÒæöæRÀÐ¢’ÓâF–7E·7G"ÂEÓ Ð¢"" Ð¢vWBÆ–W'2g&öÒF†RdÄÄÒ6öæf–ràÐ Ð¢&w3 Ð¢fÆÆÕö6öæf–s¢F†RdÄÄÒ6öæf–ràÐ¢Æ–W%÷G—S¢F†RG—RöbF†RÆ–W"FòvWBàÐ¢Æ–W%öæÖW3¢F†RæÖW2öbF†RÆ–W'2FòvWBâ–bæöæRÂ&WGW&âÆÂÆ–W'2àÐ¢"" Ð Ð¢–bÆ–W%öæÖW2—2æöæS Ð¢Æ–W%öæÖW2ÒÆ—7B‡fÆÆÕö6öæf–ræ6ö×–ÆF–öåö6öæf–rç7FF–5öf÷'v&Eö6öçFW‡Bæ¶W—2‚’Ð Ð¢f÷'v&Eö6öçFW‡BÒfÆÆÕö6öæf–ræ6ö×–ÆF–öåö6öæf–rç7FF–5öf÷'v&Eö6öçFW‡@Ð Ð¢&WGW&â°Ð¢Æ–W%öæÖS¢f÷'v&Eö6öçFW‡E¶Æ–W%öæÖUÐÐ¢f÷"Æ–W%öæÖR–âÆ–W%öæÖW0Ð¢–bÆ–W%öæÖR–âf÷'v&Eö6öçFW‡@Ð¢æB—6–ç7Fæ6R†f÷'v&Eö6öçFW‡E¶Æ–W%öæÖUÒÂÆ–W%÷G—RÐ¢ÐÐ