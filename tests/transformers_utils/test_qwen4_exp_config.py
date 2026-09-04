# SPDX-License-Identifier: Apache-2.0
"""Configuration contracts for the Qwen3.8-Flash-Next architecture."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig
from vllm.models.qwen4_exp.nvidia import model as qwen4_exp_model
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpConfig as ExportedQwen4ExpConfig,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def _text_config(**overrides: object) -> Qwen4ExpTextConfig:
    values: dict[str, object] = {
        "vocab_size": 128,
        "eos_token_id": 127,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "hc_count": 4,
        "hc_lowrank": 32,
        "ple_layer_ids": [2],
        "ple_embed_dim": 256,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 257,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
    }
    values.update(overrides)
    return Qwen4ExpTextConfig(**values)


def test_qwen4_exp_uses_one_canonical_config_class() -> None:
    assert ExportedQwen4ExpConfig is Qwen4ExpConfig


def test_qwen4_exp_nested_config_preserves_ple_and_qsa_layout() -> None:
    config = Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_text_config().to_dict(),
    )

    assert isinstance(config.text_config, Qwen4ExpTextConfig)
    assert config.text_config.short_conv_layer_ids == [1]
    assert config.text_config.ngram_context_len == 2
    assert config.text_config.indexer_budget == 2048
    assert config.architectures == ["Qwen4ExpForConditionalGeneration"]


def test_qwen4_exp_rejects_incomplete_qsa_config() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        _text_config(indexer_budget=None)


def test_qwen4_exp_model_state_allows_pipeline_parallel_ple() -> None:
    def init_base_state(
        state: Qwen4ExpModelState,
        vllm_config: object,
        model: object,
        encoder_cache: object,
        device: torch.device,
    ) -> None:
        del model, encoder_cache, device
        state.model_config = vllm_config.model_config
        state.max_num_reqs = 4

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=_text_config()),
        parallel_config=SimpleNamespace(pipeline_parallel_size=3),
    )
    with patch.object(MambaHybridModelState, "__init__", init_base_state):
        state = Qwen4ExpModelState(
            vllm_config,
            torch.nn.Identity(),
            None,
            torch.device("cpu"),
        )

    assert state.uses_ngram_embedding
    assert state.ngram_context.shape == (4, 2)


def test_qwen4_exp_hyper_connection_uses_model_dtype(monkeypatch) -> None:
    captured_dtypes: list[torch.dtype] = []

    class DummyModule(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            for arg in args:
                if isinstance(arg, qwen4_exp_model.HyperConnectionConfig):
                    captured_dtypes.append(arg.params_dtype)
                    break

    monkeypatch.setattr(qwen4_exp_model, "QwenGatedDeltaNetAttention", DummyModule)
    monkeypatch.setattr(qwen4_exp_model, "Qwen3NextMLP", DummyModule)
    monkeypatch.setattr(qwen4_exp_model, "GatedResidual", DummyModule)

    config = _text_config(ple_layer_ids=[], num_experts=0)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.float16),
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )

    qwen4_exp_model.Qwen4ExpDecoderLayer(
        vllm_config, "linear_attention", prefix="layers.0"
    )

    assert captured_dtypes == [torch.float16, torch.float16]
