# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for the Qwen4-Exp model implementation.

Qwen3.8-Flash-Next is implemented in :mod:`vllm.models.qwen4_exp` so the
backend-specific NVIDIA and ROCm paths can share the model contract. The
legacy ``vllm.model_executor.models.qwen4_exp`` import path remains supported
for model registry entries and downstream integrations.
"""

from vllm.models.qwen4_exp import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpNGramEmbedding",
]
