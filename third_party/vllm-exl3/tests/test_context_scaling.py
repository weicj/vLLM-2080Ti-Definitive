"""Tests for DeepSeek-V4 MLA context and unified-memory calculations."""

import pytest

from vllm_exl3.exl3 import compute_mla_kv_cache_bytes, validate_context_scaling


@pytest.mark.parametrize(
    ("context_len", "expected_gib"),
    [(65536, 1.51171875), (131072, 3.0234375), (262144, 6.046875), (1_000_000, 23.066997528076172)],
)
def test_mla_kv_cache_exact_bytes_and_gib(context_len, expected_gib):
    bytes_required = compute_mla_kv_cache_bytes(context_len)
    assert bytes_required == context_len * 43 * 576
    assert bytes_required / (1024**3) == pytest.approx(expected_gib)


def test_256k_context_fits_with_physical_safety_margin():
    result = validate_context_scaling(262144)
    assert result["fits"] is True
    assert result["safety_margin_gb"] > 18.0
    assert result["kv_cache_gb"] == pytest.approx(6.046875)


@pytest.mark.parametrize("context_len", [0, -1])
def test_non_positive_context_is_rejected(context_len):
    with pytest.raises(ValueError):
        validate_context_scaling(context_len)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_MODEL_WEIGHTS_GB", "90")
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_TOTAL_MEM_GB", "128")
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_MEM_UTIL", "0.95")
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_CHUNK_SIZE", "4096")
    result = validate_context_scaling(65536)
    assert result["usable_mem_gb"] == pytest.approx(121.6)
    assert result["available_headroom_gb"] == pytest.approx(30.08828125)
    assert result["chunk_size"] == 4096


def test_invalid_environment_values_keep_arguments(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_MODEL_WEIGHTS_GB", "not-a-number")
    monkeypatch.setenv("VLLM_EXL3_CONTEXT_CHUNK_SIZE", "0")
    result = validate_context_scaling(65536, model_weights_gb=90.0, chunk_size=1024)
    assert result["available_headroom_gb"] == pytest.approx(23.68828125)
    assert result["chunk_size"] == 1024
