# SPDX-License-Identifier: Apache-2.0
"""QSA launch-profile coverage for pre-Ampere and newer NVIDIA GPUs."""

import pytest

from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_sparse_launch_profile


@pytest.mark.parametrize(
    ("block_m", "base_programs", "is_pre_ampere", "expected"),
    [
        pytest.param(8, 8, True, (16, 64, 4), id="pre_ampere_small_block_m8"),
        pytest.param(16, 4, True, (16, 64, 4), id="pre_ampere_small_block_m16"),
        pytest.param(16, 5, True, (16, 32, 4), id="pre_ampere_narrow"),
        pytest.param(8, 256, True, (16, 8, 4), id="pre_ampere_split8"),
        pytest.param(8, 512, True, (16, 4, 4), id="pre_ampere_split4"),
        pytest.param(8, 513, True, (16, 1, 4), id="pre_ampere_split1"),
        pytest.param(8, 512, False, (64, 4, 2), id="ampere_split4"),
        pytest.param(8, 8192, False, (64, 1, 2), id="ampere_split1"),
    ],
)
def test_qsa_sparse_launch_profile(
    block_m: int,
    base_programs: int,
    is_pre_ampere: bool,
    expected: tuple[int, int, int],
) -> None:
    assert (
        _qsa_sparse_launch_profile(base_programs, block_m, is_pre_ampere)
        == expected
    )
