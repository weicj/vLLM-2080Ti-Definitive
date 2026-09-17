# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds checks for the align-mode Mamba state migration.

``collect_mamba_copy_meta`` hands raw device addresses to ``batch_memcpy``,
which CUDA does not bounds check. A stale block index (for example
``cur_block_idx + num_accepted_tokens - 1`` walking past the request's block
list after a preemption/resume) therefore used to become an unmapped-page
write, i.e. ``Xid 31 ... ACCESS_TYPE_VIRT_WRITE`` and a dead engine.

The bound has to be the tensor's *address span*, not ``numel()``: with
``mamba_cache_mode=align`` the state cache is page-padded. A Qwen3.8-27B conv
state is ``shape=(185, 6, 5120)`` with ``stride=(819200, 5120, 1)``, so
``numel()`` is 5,683,200 while the last block starts at offset 150,732,800.
Bounding against ``numel()`` rejects every valid migration in that layout,
which is exactly what this file's padded-layout case guards against.

Plain functions plus a ``__main__`` runner: the runtime venv ships no pytest.
"""

import torch


def test_mamba_state_copy_helpers_reject_out_of_range() -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import _block_idx_is_valid
    from vllm.v1.worker.mamba_utils import _state_range_fits

    state = torch.empty(4, 8)
    assert _block_idx_is_valid(state, [1, 2, 3], 0)
    assert _block_idx_is_valid(state, [1, 2, 3], 2)
    assert not _block_idx_is_valid(state, [1, 2, 3], 3)
    assert not _block_idx_is_valid(state, [1, 9, 3], 1)
    assert not _block_idx_is_valid(state, [1, -1, 3], 1)
    assert not _block_idx_is_valid(state, [1, 2, 3], -1)

    span = state.numel()
    assert _state_range_fits(state, state.data_ptr(), span)
    assert not _state_range_fits(state, state.data_ptr(), span + 1)
    assert _state_range_fits(state, state[3].data_ptr(), state[0].numel())
    assert not _state_range_fits(state, state[3].data_ptr(), state[0].numel() + 1)
    assert not _state_range_fits(state, 0, 0)
    assert not _state_range_fits(state, state.data_ptr(), -1)


def test_mamba_address_span_covers_padded_block_stride() -> None:
    from vllm.v1.worker.mamba_utils import (
        _state_address_span,
        _state_range_fits,
    )

    # Exactly the layout observed on the dual-2080Ti Qwen3.8-27B run:
    # 30,720 elements per block, 819,200 element stride between blocks.
    num_blocks, block_elems, block_stride = 185, 6 * 5120, 819200
    storage = torch.empty(num_blocks * block_stride, dtype=torch.float16)
    state = torch.as_strided(
        storage,
        size=(num_blocks, 6, 5120),
        stride=(block_stride, 5120, 1),
    )
    assert state.numel() == num_blocks * block_elems
    assert state.numel() < (num_blocks - 1) * block_stride

    span = _state_address_span(state)
    assert span == 1 + (num_blocks - 1) * block_stride + block_elems - 1

    # Every block, including the last, must be accepted.
    for block_id in (0, 1, 40, 42, num_blocks - 1):
        assert _state_range_fits(
            state, state[block_id].data_ptr(), state[block_id].numel()
        ), f"block {block_id} must fit"
    # One element past the last block must not.
    last = state[num_blocks - 1]
    assert not _state_range_fits(state, last.data_ptr(), last.numel() + 1)
    assert not _state_range_fits(state, 0, 0)


def test_mamba_contiguous_span_equals_numel() -> None:
    from vllm.v1.worker.mamba_utils import _state_address_span

    assert _state_address_span(torch.empty(4, 8)) == 32
    assert _state_address_span(torch.empty(0, 8)) == 0
    assert _state_address_span(torch.empty(())) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
