"""CPU-only unit tests for native-pack support: padding, n-gram row geometry,
config validation, codebook marker checks, opaque-op registration, and the
dense linear weight loader's shard-span handling.
"""

import importlib.util
import os

import pytest
import torch

pytest.importorskip("vllm")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXL3_PATH = os.path.join(_HERE, "..", "src", "vllm_exl3", "exl3.py")

_spec = importlib.util.spec_from_file_location("_exl3_native_pack_unit", _EXL3_PATH)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def test_pad128():
    assert X._exl3_pad128(4304) == 4352
    assert X._exl3_pad128(2560) == 2560


def test_ngram_words_per_row():
    assert X.ngram_words_per_row(5) == 51
    assert X.ngram_words_per_row(3) == 31


def test_ngram_embedding_config_and_lookup():
    cfg = X.Exl3Config(
        bits=3,
        codebook="mul1",
        ngram_embedding={
            "bits": 5,
            "num_shards": 4,
            "rows_per_shard": 1024,
            "num_heads": 2,
            "modules": ["ngram_embedding"],
        },
    )
    spec = cfg._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
    assert spec is not None
    assert spec["bits"] == 5

    assert cfg._ngram_embedding_spec("model.layers.1.ple.kv_proj") is None

    with pytest.raises(ValueError):
        X.Exl3Config(bits=3, codebook="mcg", ngram_embedding={"bits": 0})


def test_native_tensor_storage_maps_qwen35_fused_layers():
    cfg = X.Exl3Config.from_config(
        {
            "quant_method": "exl3",
            "codebook": "mul1",
            "tensor_storage": {
                "model.language_model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_format": "exl3",
                    "bits_per_weight": 4,
                },
                "model.language_model.layers.0.linear_attn.in_proj_z": {
                    "quant_format": "exl3",
                    "bits_per_weight": 4,
                },
                "model.language_model.layers.0.mlp.gate_proj": {
                    "quant_format": "exl3",
                    "bits_per_weight": 3,
                },
                "model.language_model.layers.0.mlp.up_proj": {
                    "quant_format": "exl3",
                    "bits_per_weight": 3,
                },
                "model.language_model.layers.0.mlp.down_proj": {
                    "quant_format": "exl3",
                    "bits_per_weight": 5,
                },
                "model.language_model.layers.0.input_layernorm": {
                    "stored_tensors": {},
                },
                "lm_head": {
                    "quant_format": "exl3",
                    "bits_per_weight": 6,
                },
            },
        }
    )

    qkvz = "language_model.model.layers.0.linear_attn.in_proj_qkvz"
    gate_up = "language_model.model.layers.0.mlp.gate_up_proj"
    down = "language_model.model.layers.0.mlp.down_proj"
    lm_head = "language_model.lm_head"
    assert cfg._matches_non_routed_exl3(qkvz)
    assert cfg._bits_for_non_routed(qkvz) == 4
    assert cfg._bits_for_non_routed(gate_up) == 3
    assert cfg._bits_for_non_routed(down) == 5
    assert cfg._bits_for_non_routed(lm_head) == 6
    assert not cfg._matches_non_routed_exl3(
        "language_model.model.layers.0.input_layernorm"
    )


def test_native_tensor_storage_rejects_mixed_fused_bit_widths():
    with pytest.raises(ValueError, match="same bit width"):
        X.Exl3Config.from_config(
            {
                "quant_method": "exl3",
                "tensor_storage": {
                    "model.language_model.layers.0.mlp.gate_proj": {
                        "quant_format": "exl3",
                        "bits_per_weight": 3,
                    },
                    "model.language_model.layers.0.mlp.up_proj": {
                        "quant_format": "exl3",
                        "bits_per_weight": 4,
                    },
                },
            }
        )


def test_streamed_ngram_disables_vocab_parallelism(monkeypatch):
    cfg = X.Exl3Config(
        bits=3,
        codebook="mul1",
        ngram_embedding={
            "bits": 5,
            "num_shards": 2,
            "rows_per_shard": 32,
            "num_heads": 2,
        },
    )
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"

    monkeypatch.delenv("VLLM_EXL3_NGRAM_STREAM", raising=False)
    assert not cfg.disable_embedding_tensor_parallel(prefix)

    monkeypatch.setenv("VLLM_EXL3_NGRAM_STREAM", "1")
    assert cfg.disable_embedding_tensor_parallel(prefix)
    assert not cfg.disable_embedding_tensor_parallel("model.layers.1.ple.key_proj")


def test_streamed_ngram_lookup_deduplicates_ids(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_NGRAM_STREAM", "1")
    cfg = X.Exl3Config(
        bits=3,
        codebook="mul1",
        ngram_embedding={
            "bits": 1,
            "num_shards": 2,
            "rows_per_shard": 4,
            "num_heads": 2,
        },
    )
    method = X.Exl3EmbeddingMethod(cfg, cfg.ngram_embedding)
    method.kernel = "torch"
    layer = torch.nn.Module()
    layer._exl3_ngram_streamed = True
    layer._exl3_ngram_dtype = torch.float16
    layer._exl3_ngram_head_offsets = torch.tensor([0, 4], dtype=torch.int64)
    layer._exl3_ngram_head_bias = torch.zeros(2, X.NGRAM_ROW_DIM, dtype=torch.float16)
    layer._exl3_ngram_stream_tables = [
        torch.zeros(4, method.words, dtype=torch.int16),
        torch.zeros(4, method.words, dtype=torch.int16),
    ]
    # Give each table row a distinct fp16 scale word. Duplicate IDs must reuse
    # the same decoded result after the unique-row disk gather.
    for shard in layer._exl3_ngram_stream_tables:
        shard[:, 0] = torch.arange(4, dtype=torch.int16)

    ids = torch.tensor([[1, 5, 1]], dtype=torch.int64)
    output = method._embedding_impl(layer, ids)

    assert output.shape == (1, 3, X.NGRAM_ROW_DIM)
    assert torch.equal(output[0, 0], output[0, 2])


def test_check_moe_codebook_markers():
    n = 4
    zeros = torch.zeros(n, 1, dtype=torch.int32)
    mul1_all = torch.full((n, 1), X.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)
    mcg_all = torch.full((n, 1), X.MCG_MARKER_SIGNED_INT32, dtype=torch.int32)

    # All mul1: ok.
    X._check_moe_codebook_markers(zeros, mul1_all, "test")
    # All mcg: ok.
    X._check_moe_codebook_markers(mcg_all, zeros, "test")
    # Neither set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(zeros, zeros, "test")
    # Both set: raises.
    with pytest.raises(RuntimeError):
        X._check_moe_codebook_markers(mcg_all, mul1_all, "test")


def test_register_opaque_layer():
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.gate_proj"
    name = X._exl3_register_opaque_layer(layer, "linear")
    assert name == f"exl3_linear:{layer.prefix}"


def test_dense_loader_span_split():
    cfg = X.Exl3Config(bits=2, codebook="mcg")
    m = X.Exl3LinearMethod(cfg, bits=2)

    layer = torch.nn.Module()
    layer.tp_rank = 0
    layer.tp_size = 1

    loader = m._make_weight_loader("svh", 3, [16, 16, 32], False, [], layer, False)

    param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param, torch.arange(64, dtype=torch.float16), (0, 1, 2))
    assert torch.equal(param.data, torch.arange(64, dtype=torch.float16))

    param2 = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    loader(param2, torch.arange(64, dtype=torch.float16), None)
    assert torch.equal(param2.data, torch.arange(64, dtype=torch.float16))

    bad_param = torch.nn.Parameter(torch.zeros(64, dtype=torch.float16), requires_grad=False)
    with pytest.raises(RuntimeError):
        loader(bad_param, torch.arange(10, dtype=torch.float16), (0, 1))
