# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.config import VllmConfig


def _verify_connector(
    monkeypatch: pytest.MonkeyPatch,
    connector: str,
    extra_config: dict[str, str] | None = None,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = object.__new__(VllmConfig)
    object.__setattr__(
        config,
        "kv_transfer_config",
        SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config=extra_config or {},
        ),
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(enable_sleep_mode=False),
    )
    config._verify_kv_transfer_compat()


@pytest.mark.parametrize(
    "connector", ["OffloadingConnector", "SimpleCPUOffloadConnector"]
)
def test_expandable_segments_allows_local_cpu_offload(
    monkeypatch: pytest.MonkeyPatch, connector: str
) -> None:
    _verify_connector(monkeypatch, connector)


def test_expandable_segments_still_rejects_registered_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="NixlConnector"):
        _verify_connector(monkeypatch, "NixlConnector")


def test_expandable_segments_rejects_custom_offloading_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="OffloadingConnector"):
        _verify_connector(
            monkeypatch,
            "OffloadingConnector",
            {"spec_name": "CustomOffloadingSpec"},
        )


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ("PYTORCH_ALLOC_CONF", "max_split_size_mb:64, expandable_segments=true"),
    ],
)
def test_expandable_segments_detection_uses_both_allocator_env_names(
    monkeypatch: pytest.MonkeyPatch, env_name: str, value: str
) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv(env_name, value)

    assert envs.is_expandable_segments_enabled()
