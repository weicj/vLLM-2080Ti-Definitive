# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import (
    default_loader,
    get_model_loader,
    register_model_loader,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


@pytest.mark.parametrize(
    ("load_format", "extra_config"),
    [
        ("auto", {"enable_multithread_load": True}),
        ("fastsafetensors", {}),
        ("instanttensor", {}),
    ],
)
def test_default_loader_rejects_specialized_loader_for_exl3_ngram_streaming(
    monkeypatch: pytest.MonkeyPatch, tmp_path, load_format: str, extra_config: dict
) -> None:
    from safetensors.torch import save_file

    save_file(
        {
            "model.ngram_embedding.shard_0.trellis": torch.ones(2, 2),
        },
        str(tmp_path / "model.safetensors"),
    )
    monkeypatch.setenv("VLLM_EXL3_NGRAM_STREAM", "1")

    with pytest.raises(ValueError, match="requires the default safetensors"):
        list(
            DefaultModelLoader(
                LoadConfig(
                    load_format=load_format,
                    model_loader_extra_config=extra_config,
                )
            )._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None))
        )


@pytest.mark.parametrize(
    ("load_format", "extra_config"),
    [
        ("auto", {"enable_multithread_load": True}),
        ("fastsafetensors", {}),
        ("instanttensor", {}),
    ],
)
def test_default_loader_allows_specialized_loader_without_exl3_ngram(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    load_format: str,
    extra_config: dict,
) -> None:
    from safetensors.torch import save_file

    save_file(
        {"model.layers.0.weight": torch.ones(2, 2)},
        str(tmp_path / "model.safetensors"),
    )
    monkeypatch.setenv("VLLM_EXL3_NGRAM_STREAM", "1")
    loader = DefaultModelLoader(
        LoadConfig(
            load_format=load_format,
            model_loader_extra_config=extra_config,
        )
    )
    iterator_name = (
        "multi_thread_safetensors_weights_iterator"
        if extra_config.get("enable_multithread_load")
        else f"{load_format}_weights_iterator"
    )
    monkeypatch.setattr(
        default_loader,
        iterator_name,
        lambda *_args, **_kwargs: iter([("model.layers.0.weight", torch.ones(2, 2))]),
    )

    weights = list(
        loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None))
    )
    assert [name for name, _ in weights] == ["model.layers.0.weight"]
    assert torch.equal(weights[0][1], torch.ones(2, 2))


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)
