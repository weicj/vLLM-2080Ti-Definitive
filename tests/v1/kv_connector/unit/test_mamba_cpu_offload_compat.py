# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace


from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler



def test_hybrid_mamba_external_hit_is_aligned_and_does_not_crash():
    """External hybrid hit is block-aligned and accepted by Mamba scheduling."""
    offload_scheduler = object.__new__(OffloadingConnectorScheduler)
    offload_scheduler._sliding_window_groups = (1,)
    offload_scheduler._lookup_groups = (0, 1)
    offload_scheduler._mamba_align_size = 16
    offload_scheduler._blocks_being_loaded = None

    class _AllHitManager:
        def lookup(self, key, req_context):
            return True

    offload_scheduler.manager = _AllHitManager()
    offload_scheduler.config = SimpleNamespace(
        kv_group_configs=(
            SimpleNamespace(
                offloaded_block_size=32, sliding_window_size_in_blocks=None
            ),
            SimpleNamespace(offloaded_block_size=16, sliding_window_size_in_blocks=1),
        )
    )
    request = SimpleNamespace(num_tokens=33, request_id="unit")
    state = SimpleNamespace(
        req=request,
        req_context=None,
        num_locally_computed_tokens=0,
        group_states=(
            SimpleNamespace(offload_keys=[b"fa0", b"fa1"]),
            SimpleNamespace(offload_keys=[b"m0", b"m1", b"m2"]),
        ),
    )

    # 33-token request is reduced to the 32-token Mamba boundary.
    assert offload_scheduler._lookup(state) == 32

    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16), use_eagle=False
    )
    model_request = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=33, num_tokens=34
    )
    # External KV tokens must be accepted by the local split calculation.
    assert (
        Scheduler._mamba_block_aligned_split(
            scheduler,
            model_request,
            20,
            num_external_computed_tokens=16,
        )
        == 16
    )


def test_mamba_align_mtp_can_retain_final_cache_boundary():
    """An uncached tail supplies MTP state without dropping a full block."""
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=14,
        num_tokens=14,
    )
    legacy_scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        use_eagle=True,
        retain_mamba_align_mtp_cache_block=False,
    )
    fixed_scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        use_eagle=True,
        retain_mamba_align_mtp_cache_block=True,
    )

    assert (
        Scheduler._mamba_block_aligned_split(legacy_scheduler, request, 14) == 8
    )
    assert (
        Scheduler._mamba_block_aligned_split(fixed_scheduler, request, 14) == 12
    )


def test_hybrid_local_hits_are_touched_before_any_external_allocation():
    """External allocation must not evict another group's untouched local hit."""
    events = []

    class _Manager:
        num_cached_block = {}

        def __init__(self, group):
            self.group = group

        def add_local_computed_blocks(self, *args):
            events.append(("local", self.group))

        def allocate_external_computed_blocks(self, *args):
            events.append(("external", self.group))

    coordinator = SimpleNamespace(
        single_type_managers=(_Manager(0), _Manager(1), _Manager(2))
    )
    KVCacheCoordinator.allocate_new_computed_blocks(
        coordinator,
        "request",
        ([object()], [object()], [object()]),
        num_local_computed_tokens=16,
        num_external_computed_tokens=32,
    )

    assert events == [
        ("local", 0),
        ("local", 1),
        ("local", 2),
        ("external", 0),
        ("external", 1),
        ("external", 2),
    ]
