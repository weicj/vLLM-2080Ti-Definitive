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


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, int],
        dtype: torch.dtype,
        *,
        is_cuda: bool = True,
        device: str = "cuda:0",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = device
        self._stride = shape if not contiguous else (shape[1], 1)

    def dim(self) -> int:
        return len(self.shape)

    def stride(self) -> tuple[int, int]:
        return self._stride


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
    # SM75 GEMV installation is independent of the SM103 shape-plan table;
    # this stands in for a TP2-local projection shape.
    assert isinstance(
        root.unlisted.quant_method, qwen4_gemm.Qwen4ExpLowLatencyLinearMethod
    )


def test_sm75_dispatches_only_runtime_eligible_gemv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = torch.empty(1, 7)
    gemv = SimpleNamespace(
        is_available=lambda: True,
        gemv=lambda x, weight: sentinel,
    )
    monkeypatch.setattr(qwen4_gemm, "triton_gemv", gemv)
    monkeypatch.setattr(qwen4_gemm, "_is_sm75", lambda: True)

    x = FakeTensor((1, 3), torch.float16)
    weight = FakeTensor((7, 3), torch.float16)
    assert qwen4_gemm._triton_runtime_ok(x, weight) is True
    assert qwen4_gemm._qwen4_exp_low_latency_gemm(x, weight) is sentinel

    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((2, 3), torch.float16), weight
    ) is False
    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((1, 3), torch.bfloat16), weight
    ) is False
    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((1, 3), torch.float16, contiguous=False), weight
    ) is False
    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((1, 4), torch.float16), weight
    ) is False
    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((1, 3), torch.float16, device="cuda:1"), weight
    ) is False
    assert qwen4_gemm._triton_runtime_ok(
        FakeTensor((1, 3), torch.float16, is_cuda=False), weight
    ) is False

    monkeypatch.setattr(qwen4_gemm, "_is_sm75", lambda: False)
    assert qwen4_gemm._triton_runtime_ok(x, weight) is False
