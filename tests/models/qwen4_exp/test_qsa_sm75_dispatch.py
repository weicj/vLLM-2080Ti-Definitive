# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia.ops.qsa import _select_config


@pytest.mark.parametrize(
    ("base_programs", "is_pre_ampere", "expected_block_n", "expected_warps"),
    [
        pytest.param(512, True, 16, 4, id="sm75-large-profile"),
        pytest.param(512, False, 64, 2, id="ampere-wide-profile"),
    ],
)
def test_qsa_sparse_dispatch_is_sm75_safe(
    base_programs: int,
    is_pre_ampere: bool,
    expected_block_n: int,
    expected_warps: int,
) -> None:
    block_n, warps, _, _ = _select_config(
        num_rows=base_programs,
        num_kv_heads=1,
        use_prefill_config=False,
        num_columns=256,
        is_pre_ampere=is_pre_ampere,
    )
    assert (block_n, warps) == (expected_block_n, expected_warps)


@pytest.mark.parametrize(
    ("has_triton", "is_sm75", "expected"),
    [(False, False, False), (True, False, True), (True, True, False)],
)
def test_qsa_metadata_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    has_triton: bool,
    is_sm75: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(qsa_cache, "HAS_TRITON", has_triton)
    monkeypatch.setattr(
        qsa_cache.current_platform,
        "is_device_capability",
        lambda capability: is_sm75 and capability == 75,
    )
    assert qsa_cache._use_triton_qsa_metadata() is expected


def test_sm75_disables_fused_pre_indexer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rotary_emb = SimpleNamespace(
        rotary_dim=64,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
        is_neox_style=True,
    )
    monkeypatch.setattr(
        indexer_qsa.current_platform,
        "is_device_capability",
        lambda capability: capability == 75,
    )
    assert not indexer_qsa._supports_fused_pre_indexer(rotary_emb, 128, 1, 8)
