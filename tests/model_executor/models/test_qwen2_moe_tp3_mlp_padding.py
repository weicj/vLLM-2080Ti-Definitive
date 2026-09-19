# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.linear import (
    PaddedMergedColumnParallelLinear,
    PaddedRowParallelLinear,
)
from vllm.model_executor.models.qwen2_moe import (
    Qwen2MoeMLP,
    _padded_intermediate_size,
)


def test_tp_partition_alignment_preserves_exl3_hadamard_blocks():
    """Quantizers can require each TP partition to start at an aligned span."""

    class BlockQuantConfig:
        tp_partition_alignment = 128

    assert _padded_intermediate_size(17408, 3, BlockQuantConfig()) == 17664


def test_qwen2_moe_mlp_pads_uneven_tp3_intermediate_dimension(
    monkeypatch,
):
    """TP3 preserves logical MLP channels and leaves its physical tail empty."""
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.models.qwen2_moe as qwen2_moe
    import vllm.model_executor.parameter as parameter

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(qwen2_moe, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(qwen2_moe, "SiluAndMul", torch.nn.Identity)

    mlp = Qwen2MoeMLP(
        hidden_size=2,
        intermediate_size=10,
        hidden_act="silu",
        quant_config=None,
        prefix="test.mlp",
    )

    assert isinstance(mlp.gate_up_proj, PaddedMergedColumnParallelLinear)
    assert isinstance(mlp.down_proj, PaddedRowParallelLinear)
    assert mlp.gate_up_proj.output_partition_sizes == [4, 4]
    assert mlp.gate_up_proj.logical_output_partition_sizes == [2, 2]
    assert mlp.down_proj.input_size_per_partition == 4
    assert mlp.down_proj.logical_input_partition_physical_offset == 0

    gate = torch.arange(20, dtype=torch.float32).view(10, 2)
    up = gate + 100
    down = torch.arange(20, dtype=torch.float32).view(2, 10)
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, gate, 0)
    mlp.gate_up_proj.weight_loader(mlp.gate_up_proj.weight, up, 1)
    mlp.down_proj.weight_loader(mlp.down_proj.weight, down)

    torch.testing.assert_close(mlp.gate_up_proj.weight[:2], gate[8:10])
    torch.testing.assert_close(mlp.gate_up_proj.weight[4:6], up[8:10])
    assert not torch.count_nonzero(mlp.gate_up_proj.weight[2:4])
    assert not torch.count_nonzero(mlp.gate_up_proj.weight[6:])
    torch.testing.assert_close(mlp.down_proj.weight[:, :2], down[:, 8:10])
    assert not torch.count_nonzero(mlp.down_proj.weight[:, 2:])
