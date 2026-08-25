# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused prefix-cache regressions for hybrid Mamba and EAGLE/MTP."""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _mamba_align_spec(block_size: int) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _make_manager(
    block_size: int,
    *,
    hybrid: bool,
    use_eagle: bool,
) -> KVCacheManager:
    groups = [KVCacheGroupSpec(["full"], _full_spec(block_size))]
    if hybrid:
        groups.append(
            KVCacheGroupSpec(["mamba"], _mamba_align_spec(block_size))
        )
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        use_eagle=use_eagle,
    )


def _warm_then_lookup(
    manager: KVCacheManager,
    token_ids: list[int],
    block_size: int,
    chunks: tuple[int, ...] | None = None,
):
    first = _make_request("first", token_ids, block_size)
    computed, num_computed = manager.get_computed_blocks(first)
    assert num_computed == 0
    for num_new_tokens in chunks or (len(token_ids),):
        blocks = manager.allocate_slots(
            first, num_new_tokens, num_computed, computed
        )
        assert blocks is not None
        first.num_computed_tokens += num_new_tokens
        computed = None
        num_computed = 0
    manager.free(first)

    second = _make_request("second", token_ids, block_size)
    return manager.get_computed_blocks(second)


def test_eagle_cache_peek_capabilities_are_conservative() -> None:
    block_size = 16
    common = dict(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=2 * block_size,
    )

    assert _full_spec(block_size).supports_eagle_cache_peek
    assert SlidingWindowSpec(**common).supports_eagle_cache_peek
    assert not SlidingWindowMLASpec(**common).supports_eagle_cache_peek
    assert not _mamba_align_spec(block_size).supports_eagle_cache_peek


def test_hybrid_mamba_eagle_does_not_reuse_lookahead_state() -> None:
    block_size = 16
    token_ids = [i for i in range(4) for _ in range(block_size)] + [4] * 7
    manager = _make_manager(block_size, hybrid=True, use_eagle=True)

    computed, num_computed = _warm_then_lookup(
        manager, token_ids, block_size, (3 * block_size, block_size + 7)
    )

    assert num_computed == 3 * block_size
    assert [len(group) for group in computed.blocks] == [3, 3]


def test_hybrid_mamba_prefix_cache_without_eagle() -> None:
    block_size = 16
    token_ids = [i for i in range(4) for _ in range(block_size)] + [4] * 7
    manager = _make_manager(block_size, hybrid=True, use_eagle=False)

    computed, num_computed = _warm_then_lookup(
        manager, token_ids, block_size, (4 * block_size, 7)
    )

    assert num_computed == 4 * block_size
    assert [len(group) for group in computed.blocks] == [4, 4]


@pytest.mark.parametrize(("use_eagle", "expected_blocks"), [(False, 4), (True, 3)])
def test_full_attention_prefix_cache_eagle_regression(
    use_eagle: bool,
    expected_blocks: int,
) -> None:
    block_size = 16
    token_ids = [i for i in range(4) for _ in range(block_size)] + [4] * 7
    manager = _make_manager(block_size, hybrid=False, use_eagle=use_eagle)

    computed, num_computed = _warm_then_lookup(
        manager, token_ids, block_size
    )

    assert num_computed == expected_blocks * block_size
    assert [len(group) for group in computed.blocks] == [expected_blocks]


def test_mamba_align_prefill_split_keeps_intermediate_chunks_aligned() -> None:
    block_size = 16
    short_request = _make_request("short", [0] * (block_size + 7), block_size)
    eagle = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=True,
    )

    assert Scheduler._mamba_block_aligned_split(eagle, short_request, 5) == 0
    assert (
        Scheduler._mamba_block_aligned_split(
            eagle, short_request, block_size + 3
        )
        == block_size
    )
    assert (
        Scheduler._mamba_block_aligned_split(
            eagle, short_request, block_size + 7
        )
        == block_size + 7
    )

    long_request = _make_request("long", [0] * (3 * block_size + 7), block_size)
    no_eagle = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=False,
    )
    assert (
        Scheduler._mamba_block_aligned_split(
            no_eagle, long_request, len(long_request.prompt_token_ids)
        )
        == 3 * block_size
    )
    long_request.num_computed_tokens = 3 * block_size
    assert Scheduler._mamba_block_aligned_split(
        no_eagle, long_request, 7
    ) == 7


def test_mamba_align_eagle_split_stops_at_reusable_boundary() -> None:
    block_size = 16
    request = _make_request("cross", [0] * (3 * block_size + 7), block_size)
    request.num_computed_tokens = block_size
    eagle = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle=True,
    )

    scheduled = Scheduler._mamba_block_aligned_split(
        eagle, request, 2 * block_size - 2
    )
    assert request.num_computed_tokens + scheduled == 2 * block_size


def test_mamba_align_allocate_tolerates_required_below_allocated() -> None:
    """Regression (EXP-039 windows): with variable-width drafts (an MTP draft
    cap below the scheduler's num_spec_tokens, or the S4 gate flipping between
    2- and 16-wide drafts) plus rejection rollback, a later step can require
    FEWER Mamba blocks than the request already holds. The base allocate path
    tolerates that (num_new_blocks <= 0 -> []); align mode asserted:
    ``num_required_blocks 17 < len(req_blocks) 18`` -> engine death on first
    spec-decode traffic. Align mode must tolerate it the same way."""
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.core.block_pool import BlockPool

    block_size = 16
    spec = MambaSpec(
        block_size=block_size,
        shapes=(1, 1),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=block_size)
    mgr = MambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    rid = "req-varwidth"
    # Step 1: a wide step (e.g. 16-token draft scheduled) grows the block list.
    first = mgr.allocate_new_blocks(rid, num_tokens=18 * block_size,
                                    num_tokens_main_model=17 * block_size)
    assert first, "first allocation should create blocks"
    held = len(mgr.req_to_blocks[rid])
    # Step 2: rejection rollback / narrow draft -> requirement DROPS below the
    # held count. Must be a graceful no-op, not an AssertionError.
    out = mgr.allocate_new_blocks(rid, num_tokens=16 * block_size,
                                  num_tokens_main_model=15 * block_size)
    assert out == []
    assert len(mgr.req_to_blocks[rid]) == held  # nothing freed mid-request
