"""CPU checks for delegation semantics, helper domains, and CI packaging gates."""

from pathlib import Path
import sys
import types

import pytest
import torch

from vllm_exl3.exl3 import (
    Exl3Config,
    filter_speculative_candidates,
    validate_context_scaling,
)


def _install_quantization_stub(monkeypatch, getter):
    """Install the smallest vLLM quantization module tree for delegation tests."""
    quantization = types.ModuleType("vllm.model_executor.layers.quantization")
    quantization.get_quantization_config = getter
    layers = types.ModuleType("vllm.model_executor.layers")
    layers.quantization = quantization
    model_executor = types.ModuleType("vllm.model_executor")
    model_executor.layers = layers
    vllm = types.ModuleType("vllm")
    vllm.model_executor = model_executor
    for name, module in (
        ("vllm", vllm),
        ("vllm.model_executor", model_executor),
        ("vllm.model_executor.layers", layers),
        ("vllm.model_executor.layers.quantization", quantization),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_declared_non_routed_delegate_has_priority(monkeypatch):
    calls = []

    class Delegate:
        @classmethod
        def from_config(cls, config):
            return ("delegate", config["quant_method"])

    def getter(name):
        calls.append(name)
        return Delegate

    _install_quantization_stub(monkeypatch, getter)
    config = object.__new__(Exl3Config)
    config.non_routed_quantization = {"quant_method": "declared_fp8", "format": "source"}

    assert config._non_routed_delegate() == ("delegate", "declared_fp8")
    assert calls == ["declared_fp8"]


def test_invalid_declared_non_routed_delegate_fails_loudly(monkeypatch):
    def getter(name):
        raise KeyError(name)

    _install_quantization_stub(monkeypatch, getter)
    config = object.__new__(Exl3Config)
    config.non_routed_quantization = {"quant_method": "missing_fp8", "bits": 8}

    with pytest.raises(RuntimeError, match="missing_fp8"):
        config._non_routed_delegate()


@pytest.mark.parametrize(
    ("max_model_len", "model_weights_gb", "total_mem_gb", "mem_util"),
    [
        (1.5, 95.4, 128.0, 0.9),
        (4096, -1.0, 128.0, 0.9),
        (4096, 95.4, 0.0, 0.9),
        (4096, 95.4, 128.0, 0.0),
        (4096, 95.4, 128.0, 2.0),
    ],
)
def test_context_scaling_rejects_invalid_domains(
    max_model_len, model_weights_gb, total_mem_gb, mem_util
):
    with pytest.raises(ValueError):
        validate_context_scaling(max_model_len, model_weights_gb, total_mem_gb, mem_util)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("nan")])
def test_speculative_filter_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        filter_speculative_candidates(torch.tensor([0.9, 0.8]), threshold=threshold)


def test_speculative_filter_tensor_return_mode_keeps_scalar_on_device():
    probs = torch.tensor([0.9, 0.4, 0.8])
    mask, kept = filter_speculative_candidates(probs, threshold=0.5, return_tensor=True)

    assert mask.tolist() == [True, False, False]
    assert isinstance(kept, torch.Tensor)
    assert kept.ndim == 0
    assert kept.dtype == torch.long
    assert kept.device == probs.device
    assert kept.item() == 1


def test_license_notice_and_ci_metadata_are_checked_in():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'license-files = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]' in pyproject
    assert "push:" in workflow and "pull_request:" in workflow
    assert "branches: [ main ]" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "python -m py_compile" in workflow
    assert "python -m pytest -v" in workflow
    assert "--sdist" in workflow
    assert "https://download.pytorch.org/whl/cpu" in workflow
