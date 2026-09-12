# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _flashqla_legacy_qkv_dtype,
)


def test_flashqla_legacy_qkv_dtype_defaults_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE", raising=False)
    assert _flashqla_legacy_qkv_dtype(torch.float16) is torch.float16
    assert _flashqla_legacy_qkv_dtype(torch.float32) is torch.float32
    assert _flashqla_legacy_qkv_dtype(torch.bfloat16) is torch.float32


def test_flashqla_legacy_native_mode_falls_back_for_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE", "native")
    assert _flashqla_legacy_qkv_dtype(torch.bfloat16) is torch.float32


def test_flashqla_legacy_qkv_dtype_can_force_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE", "fp32")
    assert _flashqla_legacy_qkv_dtype(torch.float16) is torch.float32


def test_flashqla_legacy_qkv_dtype_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLASHQLA_LEGACY_GDN_QKV_DTYPE", "bf16")
    with pytest.raises(ValueError, match="must be 'fp32' or 'native'"):
        _flashqla_legacy_qkv_dtype(torch.float16)
