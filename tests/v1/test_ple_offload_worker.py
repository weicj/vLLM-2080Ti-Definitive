# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.v1.ple_offload.worker import _without_pp_layer_partition


def test_offload_model_discovery_ignores_gpu_pp_partition(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "24,24")

    with _without_pp_layer_partition():
        assert "VLLM_PP_LAYER_PARTITION" not in os.environ

    assert os.environ["VLLM_PP_LAYER_PARTITION"] == "24,24"
