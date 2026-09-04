# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.model_executor.layers import ple_offload_layer


def test_target_device_uses_gpu_in_main_worker(monkeypatch):
    monkeypatch.setattr(ple_offload_layer.envs, "VLLM_PLE_CPU_OFFLOAD", True)
    monkeypatch.setattr(ple_offload_layer, "_offload_worker_flag", False)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    assert ple_offload_layer.PleOffloadLayer.get_target_device() == torch.device(
        "cuda", 3
    )


def test_target_device_uses_cpu_only_in_offload_process(monkeypatch):
    monkeypatch.setattr(ple_offload_layer.envs, "VLLM_PLE_CPU_OFFLOAD", True)
    monkeypatch.setattr(ple_offload_layer, "_offload_worker_flag", True)

    assert ple_offload_layer.PleOffloadLayer.get_target_device() == torch.device("cpu")


def test_ple_wait_schema_orders_on_hidden_states() -> None:
    schema = torch.ops.vllm.ple_offload_wait.default._schema
    write_args = {
        arg.name
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    }

    assert write_args == {"hidden_states"}
