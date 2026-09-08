# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import low_latency_gemm as qwen4_gemm


class FakeLinear(nn.Module):
    def __init__(self, quant_method: object, n: int, k: int) -> None:
        super().__init__()
        self.quant_method = quant_method
        self.weight = nn.Parameter(torch.empty(n, k), requires_grad=False)


class FakeHead(FakeLinear):
    pass


def _make_module() -> tuple[nn.Module, object]:
    root = nn.Module()
    root.linear = FakeLinear(qwen4_gemm.UnquantizedLinearMethod(), 640, 2560)
    quantized_method = object()
    root.quantized = FakeLinear(quantized_method, 3584, 2560)
    root.unlisted = FakeLinear(qwen4_gemm.UnquantizedLinearMethod(), 123, 456)
    root.lm_head = FakeHead(qwen4_gemm.UnquantizedEmbeddingMethod(), 62080, 2560)
    return root, quantized_method


def test_sm75_installs_fp16_triton_path(monkeypatch: pytest.MonkeyPatch) -> None:
    root, quantized_method = _make_module()
    monkeypatch.setattr(qwen4_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(qwen4_gemm, "ParallelLMHead", FakeHead)
    monkeypatch.setattr(qwen4_gemm, "_is_sm75", lambda: True)
    monkeypatch.setattr(qwen4_gemm, "_is_sm103", lambda: False)
    monkeypatch.setattr(qwen4_gemm.triton_gemv, "is_available", lambda: True)

    qwen4_gemm.enable_qwen4_exp_low_latency_gemm(root, torch.float16)

    assert isinstance(root.linear.quant_method, qwen4_gemm.Qwen4ExpLowLatencyLinearMethod)
    assert isinstance(
        root.lm_head.quant_method, qwen4_gemm.Qwen4ExpLowLatencyEmbeddingMethod
    )
    assert root.quantized.quant_method is quantized_method
    assert type(root.unlisted.quant_method) is qwen4_gemm.UnquantizedLinearMethod


def test_sm75_dispatches_only_runtime_eligible_gemv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = torch.empty(1, 7)
    gemv = SimpleNamespace(
        is_available=lambda: True,
        gemv=lambda x, weight: sentinel,
    )
    monkeypatch.setattr(qwen4_gemm, "triton_gemv", gemv)
    monkeypatch.setattr(qwen4_gemm, "_triton_runtime_ok", lambda x, weight: True)

    x = torch.empty(1, 3)
    weight = torch.empty(7, 3)
    assert qwen4_gemm._qwen4_exp_low_latency_gemm(x, weight) is sentinel

    monkeypatch.setattr(qwen4_gemm, "_triton_runtime_ok", lambda x, weight: False)
    actual = qwen4_gemm._qwen4_exp_low_latency_gemm(x, weight)
    torch.testing.assert_close(actual, torch.nn.functional.linear(x, weight))
