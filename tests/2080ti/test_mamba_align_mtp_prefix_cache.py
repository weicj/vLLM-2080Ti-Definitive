# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler


def _split_with_mtp_cache_retention(enabled: bool, num_tokens: int = 14) -> int:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_eagle = True
    scheduler.retain_mamba_align_mtp_cache_block = enabled
    scheduler.cache_config = SimpleNamespace(block_size=4)
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
    )
    return scheduler._mamba_block_aligned_split(request, num_new_tokens=14)


def test_mamba_align_mtp_can_retain_final_cache_boundary() -> None:
    """MTP should not discard a complete Mamba block before the prompt tail."""
    # The legacy EAGLE rule drops the final complete block (12 -> 8).
    assert _split_with_mtp_cache_retention(enabled=False) == 8
    # MTP still computes the two-token tail, so the 12-token state is valid.
    assert _split_with_mtp_cache_retention(enabled=True) == 12
    # Without a tail, preserve EAGLE's safety block.
    assert _split_with_mtp_cache_retention(enabled=True, num_tokens=12) == 8
