# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for the SM75 FP16 QSA attention path."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qsa_sparse_paged_attention_fp16_matches_reference() -> None:
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
