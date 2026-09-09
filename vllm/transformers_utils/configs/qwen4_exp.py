# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the Qwen4-Exp configuration.

The model package owns the canonical configuration class. Keeping this module
as a re-export preserves the transformers-utils lookup path without creating a
second, incompatible config type.
"""

from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)

__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
]
