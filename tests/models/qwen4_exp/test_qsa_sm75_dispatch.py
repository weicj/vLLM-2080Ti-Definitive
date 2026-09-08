# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.nvidia import indexer_qsa


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


def test_non_sm75_keeps_fused_pre_indexer(
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
        lambda capability: False,
    )
    assert indexer_qsa._supports_fused_pre_indexer(rotary_emb, 128, 1, 8)
