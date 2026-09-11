# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLELayer,
    _disable_ple_embedding_tp,
)


def test_ple_embedding_tp_can_be_disabled_by_a_quantizer() -> None:
    class StreamedEmbeddingQuantizer:
        def disable_embedding_tensor_parallel(self, prefix: str) -> bool:
            return prefix.endswith("ngram_embedding")

    quantizer = StreamedEmbeddingQuantizer()
    assert _disable_ple_embedding_tp(quantizer, "layers.1.ngram_embedding")
    assert not _disable_ple_embedding_tp(quantizer, "layers.1.key_proj")
    assert not _disable_ple_embedding_tp(None, "layers.1.ngram_embedding")


def test_prefill_short_conv_ignores_cuda_graph_padding() -> None:
    """Padded prefill rows must not index or mutate a real request state."""
    hidden_size = 4
    layer = object.__new__(Qwen4ExpPLELayer)
    layer.conv_state_len = 2
    layer.short_conv_dilation = 1

    metadata = SimpleNamespace(
        non_spec_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        has_initial_states_p=torch.tensor([True]),
        max_prefill_query_len=2,
    )
    inputs = torch.ones((512, hidden_size))
    conv_state = torch.zeros((2, hidden_size, 2))
    conv_weights = torch.ones((hidden_size, 3))

    output = layer._short_conv_dilated_prefill_batched(
        inputs,
        metadata,
        conv_state,
        conv_weights,
        torch.tensor([1], dtype=torch.int32),
        num_prefills=1,
        num_decode_tokens=0,
        num_prefill_tokens=512,
    )

    expected = torch.tensor(
        [[torch.nn.functional.silu(torch.tensor(1.0))] * hidden_size,
         [torch.nn.functional.silu(torch.tensor(2.0))] * hidden_size]
    )
    torch.testing.assert_close(output[:2], expected)
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))
    torch.testing.assert_close(conv_state[0], torch.zeros_like(conv_state[0]))
    torch.testing.assert_close(conv_state[1], torch.ones_like(conv_state[1]))


def test_decode_short_conv_does_not_write_graph_padding_rows() -> None:
    """A padded NULL block must not overwrite a live decode state."""
    hidden_size = 4
    layer = object.__new__(Qwen4ExpPLELayer)
    layer.conv_state_len = 2
    layer.short_conv_dilation = 1

    conv_state = torch.zeros((3, hidden_size, 2))
    conv_state[1].fill_(2.0)
    conv_weights = torch.ones((hidden_size, 3))
    output = layer._short_conv_dilated_decode_batched(
        torch.ones((2, hidden_size)),
        conv_state,
        conv_weights,
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([True, False]),
    )

    assert output.shape == (2, hidden_size)
    torch.testing.assert_close(conv_state[0], torch.zeros_like(conv_state[0]))
    expected_state = torch.tensor([2.0, 1.0]).expand(hidden_size, -1)
    torch.testing.assert_close(conv_state[1], expected_state)
