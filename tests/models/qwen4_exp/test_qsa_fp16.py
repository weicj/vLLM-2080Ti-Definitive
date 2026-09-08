# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the SM75 FP16 QSA attention path."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
    qsa_store_cache_rows,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="requires NVIDIA CUDA and Triton",
)
@pytest.mark.parametrize("torch_fallback", [False, True])
def test_qsa_sparse_paged_attention_fp16_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    torch_fallback: bool,
) -> None:
    monkeypatch.setenv("VLLM_QSA_TORCH_SPARSE", str(int(torch_fallback)))
    torch.manual_seed(7)
    device = torch.device("cuda")
    page_size = 16
    head_dim = 128
    num_pages = 8
    num_query_heads = 4
    rows = 3
    selected_tokens = torch.tensor(
        [
            [0, 3, 17, 31, 47, 79],
            [2, 16, 32, 45, 63, 95],
            [4, 18, 33, 61, 80, 111],
        ],
        dtype=torch.int32,
        device=device,
    )
    query = torch.randn(
        rows, num_query_heads, head_dim, dtype=torch.float16, device=device
    )
    key_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.float16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)

    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        selected_tokens,
        block_table,
        token_to_req,
    )

    expected = torch.empty_like(actual)
    scale = head_dim**-0.5
    for row, token_ids in enumerate(selected_tokens.cpu().tolist()):
        keys = torch.stack(
            [
                key_cache[token // page_size, token % page_size, 0]
                for token in token_ids
            ]
        ).float()
        values = torch.stack(
            [
                value_cache[token // page_size, token % page_size, 0]
                for token in token_ids
            ]
        ).float()
        for head in range(num_query_heads):
            weights = torch.softmax(keys @ query[row, head].float() * scale, dim=0)
            expected[row, head] = (weights @ values).to(torch.float16)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="requires NVIDIA CUDA and Triton",
)
def test_qsa_sparse_paged_attention_flash_next_capture_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the 256-wide TP capture tile launchable on SM75."""
    monkeypatch.setenv("VLLM_QSA_TORCH_SPARSE", "0")
    device = torch.device("cuda")
    rows, num_query_heads, head_dim = 512, 6, 256
    page_size, selected_tokens = 16, 256

    query = torch.randn(
        rows,
        num_query_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    key_cache = torch.randn(
        16,
        page_size,
        1,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    logical_indices = (
        torch.arange(selected_tokens, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(rows, -1)
        .contiguous()
    )
    block_table = torch.arange(16, dtype=torch.int32, device=device).unsqueeze(0)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)

    actual = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
    )
    torch.cuda.synchronize()

    assert actual.shape == query.shape
    assert torch.isfinite(actual).all()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="requires NVIDIA CUDA and Triton",
)
def test_qsa_sm75_torch_cache_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_QSA_TORCH_CACHE", "1")
    device = torch.device("cuda")

    cache = torch.zeros((2, 4, 1, 8), dtype=torch.float16, device=device)
    rows = torch.arange(32, dtype=torch.float16, device=device).reshape(4, 8)
    slots = torch.tensor([1, 5, -1, 8], dtype=torch.int64, device=device)
    qsa_store_cache_rows(cache, slots, rows)
    torch.testing.assert_close(cache[0, 1, 0], rows[0])
    torch.testing.assert_close(cache[1, 1, 0], rows[1])
    assert torch.count_nonzero(cache).item() == 15

    raw_keys = (
        torch.arange(8, dtype=torch.float16, device=device)
        .reshape(8, 1, 1)
        .expand(-1, 1, 8)
        .contiguous()
    )
    raw_positions = (
        torch.arange(8, dtype=torch.int64, device=device)
        .reshape(8, 1, 1)
        .expand(-1, 1, 3)
        .contiguous()
    )
    state_cache = torch.zeros((1, 8, 1, 8), dtype=torch.float16, device=device)
    state_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    token_to_req = torch.zeros(8, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device=device)
    logical_positions = torch.arange(8, dtype=torch.int64, device=device)
    compressed_slots = torch.tensor(
        [-1, -1, -1, 0, -1, -1, -1, 1], dtype=torch.int64, device=device
    )
    pooled, first_positions = qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        state_cache,
        state_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        4,
    )
    expected = torch.zeros_like(pooled)
    expected[3] = 1.5
    expected[7] = 5.5
    torch.testing.assert_close(pooled, expected)
    assert first_positions[3].tolist() == [0, 0, 0]
    assert first_positions[7].tolist() == [4, 4, 4]


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="requires NVIDIA CUDA and Triton",
)
def test_qsa_sm75_torch_topk_matches_torch_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    rows, num_heads, head_dim = 4, 6, 128
    page_size, num_pages = 16, 8
    query = torch.randn(
        rows, num_heads, head_dim, dtype=torch.float16, device=device
    )
    key_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.float16, device=device
    )
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).unsqueeze(0)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    query_positions = torch.tensor(
        [31, 63, 95, 127], dtype=torch.int64, device=device
    )
    sequence_lengths = torch.tensor([1024], dtype=torch.int32, device=device)

    monkeypatch.setenv("VLLM_QSA_TORCH_SPARSE", "1")
    expected = qsa_select_paged_tokens(
        query,
        key_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk=32,
        compress_ratio=8,
    )
    monkeypatch.setenv("VLLM_QSA_TORCH_SPARSE", "0")
    monkeypatch.setenv("VLLM_QSA_TORCH_TOPK", "1")
    actual = qsa_select_paged_tokens(
        query,
        key_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk=32,
        compress_ratio=8,
    )
    torch.testing.assert_close(actual, expected)
