# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)


def test_qwen4_exp_config_normalizes_legacy_vision_type():
    config = Qwen4ExpConfig(
        text_config={
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [],
        },
        vision_config={"model_type": "qwen4_exp", "out_hidden_size": 2560},
    )

    assert isinstance(config.text_config, Qwen4ExpTextConfig)
    assert config.text_config.layer_types[-1] == "full_attention"
    assert config.text_config.output_gate_type == "silu"
    assert config.vision_config.model_type == "qwen4_exp_vision"
    config.text_config.validate_architecture()


def test_qwen4_exp_qsa_requires_a_complete_tuple():
    config = Qwen4ExpTextConfig(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=None,
        indexer_compress_ratio=4,
        ple_layer_ids=[],
    )

    with pytest.raises(ValueError, match="QSA requires all indexer fields"):
        config.validate_architecture()


def test_qwen4_exp_rejects_layer_type_length_mismatch():
    with pytest.raises(ValueError, match="one entry per layer"):
        Qwen4ExpTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention"],
            ple_layer_ids=[],
        )
