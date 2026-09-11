"""Unit tests for adaptive verification head confidence filtering and pruning."""

import pytest

from vllm_exl3.exl3 import (
    is_adaptive_verification_enabled,
    filter_speculative_candidates,
)

try:
    import torch
except ImportError:
    torch = None


def test_adaptive_verification_env_toggle(monkeypatch):
    """Verify environment variable toggle for adaptive verification."""
    monkeypatch.delenv("VLLM_EXL3_ADAPTIVE_VERIFICATION", raising=False)
    assert not is_adaptive_verification_enabled()

    monkeypatch.setenv("VLLM_EXL3_ADAPTIVE_VERIFICATION", "1")
    assert is_adaptive_verification_enabled()

    monkeypatch.setenv("VLLM_EXL3_ADAPTIVE_VERIFICATION", "true")
    assert is_adaptive_verification_enabled()

    monkeypatch.setenv("VLLM_EXL3_ADAPTIVE_VERIFICATION", "0")
    assert not is_adaptive_verification_enabled()


def test_confidence_pruning_all_confident():
    """When all candidate tokens exceed threshold, all are kept."""
    if torch is None:
        pytest.skip("PyTorch required for tensor pruning tests")
    probs = torch.tensor([[0.9, 0.85, 0.8]])
    mask, num_tokens = filter_speculative_candidates(probs, threshold=0.7)
    assert num_tokens == 3
    assert mask.shape == probs.shape
    assert mask.all()


def test_confidence_pruning_early_cutoff():
    """When a middle token drops below threshold, subsequent tokens are pruned."""
    if torch is None:
        pytest.skip("PyTorch required for tensor pruning tests")
    probs = torch.tensor([[0.9, 0.4, 0.85]])
    mask, num_tokens = filter_speculative_candidates(probs, threshold=0.7)
    assert num_tokens == 1
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False
    assert mask[0, 2].item() is False


def test_confidence_pruning_batch():
    """Verify independent candidate pruning per batch sequence."""
    if torch is None:
        pytest.skip("PyTorch required for tensor pruning tests")
    probs = torch.tensor([
        [0.95, 0.90, 0.80],
        [0.50, 0.90, 0.90]
    ])
    mask, counts = filter_speculative_candidates(probs, threshold=0.7)
    assert counts.tolist() == [3, 0]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [False, False, False]


def test_confidence_pruning_edge_cases():
    """Test empty edge cases."""
    if torch is None:
        pytest.skip("PyTorch required for tensor pruning tests")
    probs = torch.empty((1, 0))
    mask, count = filter_speculative_candidates(probs, threshold=0.5)
    assert count == 0
