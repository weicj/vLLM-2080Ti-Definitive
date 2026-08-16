# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # None below means full attention
    sliding_window_size_in_blocks: int | None


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(spec: "OffloadingSpec") -> int | None:
    """Return the offloaded token boundary required by Mamba align mode."""
    mamba_align_size: int | None = None
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            offload_block_size = gpu_block_size * spec.block_size_factor
            assert mamba_align_size is None or mamba_align_size == offload_block_size
            mamba_align_size = offload_block_size
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=get_sliding_window_size_in_blocks(
                        spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                        gpu_block_size * spec.block_size_factor,
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    req_context: ReqContext = field(init=False)
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        self.req_context = ReqContext(kv_transfer_params=self.req.kv_transfer_params)

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec)

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            # A block can be reallocated and its pending-store entry flushed
            # before the worker completion arrives. Completion is still
            # valid; do not take down EngineCore while cleaning up that stale
            # bookkeeping entry.
            pending = self._block_id_to_pending_jobs.get(bid)
            if pending is None:
                continue
            pending.discard(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
        if self._mamba_align_size is not None:
            # Mamba align stores a single recurrent state at block boundaries.
            # Never load a partial boundary or a state beyond the valid hit.
            max_hit_size_tokens = round_down(
                max_hit_size_tokens, self._mamba_align_size
            )
        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfigßm7¶‰žËkºwµç@€€¹Õµ}±½…±±å}½µÁÕÑ•‘}ÁÕ}‰±½­Ìé¹Õµ}ÁÕ}‰±½­Ì4(€€€€€€€€€€€€€€€t4(€€€€€€€€€€€€¤4(€€€€€€€€€€€É½ÕÁ}Í¥é•Ì¹…ÁÁ•¹¡¹Õµ}Á•¹‘¥¹}ÁÕ}‰±½­Ì¤4(€€€€€€€€€€€‰±½­}¥¹‘¥•Ì¹…ÁÁ•¹¡¹Õµ}±½…±±å}½µÁÕÑ•‘}ÁÕ}‰±½­Ì¤4(4(€€€€€€€€€€€¥˜¹½Ð‘½}É•µ½Ñ•}‘•½‘”è4(€€€€€€€€€€€€€€€€Œ½È@½ÁÉ•™¥±°É•ÅÕ•ÍÑÌ€¡‘½}É•µ½Ñ•}‘•½‘”õQÉÕ”¤°Ý”‘¼4(€€€€€€€€€€€€€€€€Œ9=PÍ­¥ÀÍ…Ù¥¹œÑ¡”¡¥ÐÁÉ•™¥à°…ÌÝ”¹••Ñ¼ÍÑÉ•…´Ñ¡”4(€€€€€€€€€€€€€€€€Œ•¹Ñ¥É”-X…¡”Í¼„É•µ½Ñ”‘•½‘”¹½‘”…¸½¹ÍÕµ”¥Ð¸4(€€€€€€€€€€€€€€€É½ÕÁ}ÍÑ…Ñ”¹¹•áÑ}ÍÑ½É•‘}‰±½­}¥‘à€ô¹Õµ}‰±½­Ì4(4(€€€€€€€€Œ•¹”‘ÍÐ‰±½­Ì……¥¹ÍÐ™¥¹¥Í¡•µÉ•ÅÕ•ÍÐÁ•¹‘¥¹œÍÑ½É•Ì¸4(€€€€€€€¥˜€ 4(€€€€€€€€€€€Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì4(€€€€€€€€€€€…¹¹½ÐÍ•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹­•åÌ ¤¹¥Í‘¥Í©½¥¹Ð¡‘ÍÑ}‰±½­}¥‘Ì¤4(€€€€€€€€¤è4(€€€€€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}©½‰Í}Ñ½}™±ÕÍ ¹ÕÁ‘…Ñ” 4(€€€€€€€€€€€€€€€©¥4(€€€€€€€€€€€€€€€™½È‰¥¥¸‘ÍÑ}‰±½­}¥‘Ì4(€€€€€€€€€€€€€€€™½È©¥¥¸Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹•Ð¡‰¥°€ ¤¤4(€€€€€€€€€€€€¤4(4(€€€€€€€ÍÉ}ÍÁ•Œ€ôÍ•±˜¹µ…¹…•È¹ÁÉ•Á…É•}±½…¡­•åÍ}Ñ½}±½…°É•Å}ÍÑ…ÑÕÌ¹É•Å}½¹Ñ•áÐ¤4(€€€€€€€‘ÍÑ}ÍÁ•Œ€ôAU1½…‘MÑ½É•MÁ•Œ 4(€€€€€€€€€€€‘ÍÑ}‰±½­}¥‘Ì°É½ÕÁ}Í¥é•ÌõÉ½ÕÁ}Í¥é•Ì°‰±½­}¥¹‘¥•Ìõ‰±½­}¥¹‘¥•Ì4(€€€€€€€€¤4(4(€€€€€€€±½…‘}©½‰}¥€ôÍ•±˜¹}•¹•É…Ñ•}©½‰}¥ ¤4(€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}±½…‘}©½‰Ím±½…‘}©½‰}¥‘t€ôQÉ…¹Í™•É)½ˆ 4(€€€€€€€€€€€É•Å}¥õÉ•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥°4(€€€€€€€€€€€ÑÉ…¹Í™•É}ÍÁ•Œô¡ÍÉ}ÍÁ•Œ°‘ÍÑ}ÍÁ•Œ¤°4(€€€€€€€€¤4(€€€€€€€€Œ„±½……¸½¹±ä‰”¥ÍÍÕ•Ý¡•¸¹¼½Ñ¡•È©½‰Ì…É”Á•¹‘¥¹œ¸4(€€€€€€€…ÍÍ•ÉÐ¹½ÐÉ•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì4(€€€€€€€É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¹…‘¡±½…‘}©½‰}¥¤4(€€€€€€€Í•±˜¹}©½‰Ím±½…‘}©½‰}¥‘t€ôQÉ…¹Í™•É)½‰MÑ…ÑÕÌ 4(€€€€€€€€€€€É•Å}¥õÉ•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥°4(€€€€€€€€€€€Á•¹‘¥¹}½Õ¹ÐõÍ•±˜¹½¹™¥œ¹¹Õµ}Ý½É­•ÉÌ°4(€€€€€€€€€€€­•åÌõÍ•Ð¡­•åÍ}Ñ½}±½…¤°4(€€€€€€€€€€€¥Í}ÍÑ½É”õ…±Í”°4(€€€€€€€€¤4(4(€€€€€€€¥˜Í•±˜¹}‰±½­Í}‰•¥¹}±½…‘•¥Ì¹½Ð9½¹”è4(€€€€€€€€€€€Í•±˜¹}‰±½­Í}‰•¥¹}±½…‘•¹ÕÁ‘…Ñ”¡­•åÍ}Ñ½}±½…¤4(4(€€€‘•˜}‰Õ¥±‘}ÍÑ½É•}©½‰Ì 4(€€€€€€€Í•±˜°4(€€€€€€€Í¡•‘Õ±•É}½ÕÑÁÕÐèM¡•‘Õ±•É=ÕÑÁÕÐ°4(€€€€¤€´ø‘¥Ñm¥¹Ð°QÉ…¹Í™•É)½‰tè4(€€€€€€€‰±½­}Í¥é•}™…Ñ½È€ôÍ•±˜¹½¹™¥œ¹‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€ÍÑ½É•}©½‰Ìè‘¥Ñm¥¹Ð°QÉ…¹Í™•É)½‰t€ôíô4(€€€€€€€€Œ¥Ñ•É…Ñ”½Ù•È‰½Ñ ¹•Ü…¹…¡•É•ÅÕ•ÍÑÌ4(€€€€€€€™½ÈÉ•Å}¥°¹•Ý}‰±½­}¥‘}É½ÕÁÌ°ÁÉ••µÁÑ•¥¸å¥•±‘}É•Å}‘…Ñ„¡Í¡•‘Õ±•É}½ÕÑÁÕÐ¤è4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ€ôÍ•±˜¹}É•Å}ÍÑ…ÑÕÍmÉ•Å}¥‘t4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ•}½™™±½…‘}­•åÌ ¤4(€€€€€€€€€€€É•Ä€ôÉ•Å}ÍÑ…ÑÕÌ¹É•Ä4(4(€€€€€€€€€€€¥˜ÁÉ••µÁÑ•è4(€€€€€€€€€€€€€€€™½ÈÉ½ÕÁ}ÍÑ…Ñ”¥¸É•Å}ÍÑ…ÑÕÌ¹É½ÕÁ}ÍÑ…Ñ•Ìè4(€€€€€€€€€€€€€€€€€€€É½ÕÁ}ÍÑ…Ñ”¹‰±½­}¥‘Ì¹±•…È ¤4(4(€€€€€€€€€€€¥˜¹•Ý}‰±½­}¥‘}É½ÕÁÌè4(€€€€€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹ÕÁ‘…Ñ•}‰±½­}¥‘}É½ÕÁÌ¡¹•Ý}‰±½­}¥‘}É½ÕÁÌ¤4(€€€€€€€€€€€€€€€€Œ•¹”¹•Ü‰±½­Ì……¥¹ÍÐ¥¸µ™±¥¡ÐÍÑ½É•Ì¸4(€€€€€€€€€€€€€€€¥˜Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ìè4(€€€€€€€€€€€€€€€€€€€¹•Ý}‰±½­Í}™±…Ð€ôl4(€€€€€€€€€€€€€€€€€€€€€€€‰¥™½È¹•Ý}‰±½­Ì¥¸¹•Ý}‰±½­}¥‘}É½ÕÁÌ™½È‰¥¥¸¹•Ý}‰±½­Ì4(€€€€€€€€€€€€€€€€€€€t4(€€€€€€€€€€€€€€€€€€€¥˜¹½ÐÍ•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹­•åÌ ¤¹¥Í‘¥Í©½¥¹Ð 4(€€€€€€€€€€€€€€€€€€€€€€€¹•Ý}‰±½­Í}™±…Ð4(€€€€€€€€€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}©½‰Í}Ñ½}™±ÕÍ ¹ÕÁ‘…Ñ” 4(€€€€€€€€€€€€€€€€€€€€€€€€€€€©¥4(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½È‰¥¥¸¹•Ý}‰±½­Í}™±…Ð4(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½È©¥¥¸Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹•Ð¡‰¥°€ ¤¤4(€€€€€€€€€€€€€€€€€€€€€€€€¤4(4(€€€€€€€€€€€¹Õµ}Í¡•‘Õ±•‘}Ñ½­•¹Ì€ôÍ¡•‘Õ±•É}½ÕÑÁÕÐ¹¹Õµ}Í¡•‘Õ±•‘}Ñ½­•¹ÍmÉ•Å}¥‘t4(€€€€€€€€€€€¹Õµ}Ñ½­•¹Í}…™Ñ•É}‰…Ñ €ôÉ•Ä¹¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì€¬¹Õµ}Í¡•‘Õ±•‘}Ñ½­•¹Ì4(€€€€€€€€€€€€ŒÝ¥Ñ …Íå¹ŒÍ¡•‘Õ±¥¹œ°Í½µ”Ñ½­•¹Ìµ…ä‰”µ¥ÍÍ¥¹œ4(€€€€€€€€€€€¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì€ôµ¥¸¡¹Õµ}Ñ½­•¹Í}…™Ñ•É}‰…Ñ °É•Ä¹¹Õµ}Ñ½­•¹Ì¤4(4(€€€€€€€€€€€€Œ¥±Ñ•È½ÕÐ‰±½­ÌÍ­¥ÁÁ•‘Õ”Ñ¼Í±¥‘¥¹œÝ¥¹‘½Ü…ÑÑ•¹Ñ¥½¸€¼MM44(€€€€€€€€€€€¹•Ý}½™™±½…‘}­•åÌè±¥ÍÑm=™™±½…‘-•åt€ômt4(€€€€€€€€€€€™½ÈÉ½ÕÁ}½¹™¥œ°É½ÕÁ}ÍÑ…Ñ”¥¸é¥À 4(€€€€€€€€€€€€€€€Í•±˜¹½¹™¥œ¹­Ù}É½ÕÁ}½¹™¥Ì°É•Å}ÍÑ…ÑÕÌ¹É½ÕÁ}ÍÑ…Ñ•Ì4(€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€¹Õµ}‰±½­Ì€ô¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì€¼¼É½ÕÁ}½¹™¥œ¹½™™±½…‘•‘}‰±½­}Í¥é”4(€€€€€€€€€€€€€€€ÍÑ…ÉÑ}‰±½­}¥‘à€ôÉ½ÕÁ}ÍÑ…Ñ”¹¹•áÑ}ÍÑ½É•‘}‰±½­}¥‘à4(€€€€€€€€€€€€€€€¥˜¹Õµ}‰±½­Ì€ðôÍÑ…ÉÑ}‰±½­}¥‘àè4(€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€€€€€½™™±½…‘}­•åÌ€ôÉ½ÕÁ}ÍÑ…Ñ”¹½™™±½…‘}­•åÍmÍÑ…ÉÑ}‰±½­}¥‘àé¹Õµ}‰±½­Ít4(€€€€€€€€€€€€€€€€Œ½È•… ‰±½¬Ñ¼½™™±½…°Ñ…­”Ñ¡”±…ÍÐ½ÉÉ•ÍÁ½¹‘¥¹œAT‰±½¬¸4(€€€€€€€€€€€€€€€€Œ”¹œ¸¥˜‰±½¬Í¥é”™…Ñ½È¥Ì€Ì…¹AT‰±½¬%Ì…É”4(€€€€€€€€€€€€€€€€Œ€Ä€Ô€Ø€Ü€È€Ð€ä€Ì€àÑ¡•¸Ý”±°Ñ…­”‰±½­Ì€Ø€Ð€à¸4(€€€€€€€€€€€€€€€€Œ]”Ý¥±°ÕÍ”Ñ¡•Í”AT‰±½­ÌÑ¼‘•Ñ•Éµ¥¹”¥˜Ñ¡”‰±½¬¹••‘Ì4(€€€€€€€€€€€€€€€€Œ½™™±½…‘¥¹œ°½È€¡¥˜Ñ¡”AT‰±½¬%¥Ì€À¤Ñ¡¥Ì‰±½¬Í¡½Õ±4(€€€€€€€€€€€€€€€€Œ‰”Í­¥ÁÁ•‘Õ”Ñ¼Í±¥‘¥¹œÝ¥¹‘½Ü…ÑÑ•¹Ñ¥½¸€¼MM4¸4(€€€€€€€€€€€€€€€€Œ]”­¹½ÜÑ¡…Ð¥˜„‰±½¬¥ÌÍ­¥ÁÁ•°Ñ¡•¸…±°Ñ¡”ÁÉ•Ù¥½ÕÌ‰±½­Ì4(€€€€€€€€€€€€€€€€Œ…É”Í­¥ÁÁ•…ÌÝ•±°¸Q¡¥Ì¥ÌÝ¡äÝ”Ñ…­”Ñ¡”±…ÍÐ½˜•… ‰±½¬¸4(€€€€€€€€€€€€€€€½™™±½…‘}‰±½­}¥‘Ì€ôÉ½ÕÁ}ÍÑ…Ñ”¹‰±½­}¥‘Íl4(€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ}‰±½­}¥‘à€¨‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€€€€€€€€€€€€€€¬‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€€€€€€€€€€€€€€´€Ä€è¹Õµ}‰±½­Ì€¨‰±½­}Í¥é•}™…Ñ½È€è‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€€€€€€€€€t4(€€€€€€€€€€€€€€€…ÍÍ•ÉÐ±•¸¡½™™±½…‘}­•åÌ¤€ôô±•¸¡½™™±½…‘}‰±½­}¥‘Ì¤4(4(€€€€€€€€€€€€€€€™½È½™™±½…‘}­•ä°‰±½­}¥¥¸é¥À¡½™™±½…‘}­•åÌ°½™™±½…‘}‰±½­}¥‘Ì¤è4(€€€€€€€€€€€€€€€€€€€¥˜‰±½­}¥€„ô€Àè4(€€€€€€€€€€€€€€€€€€€€€€€¹•Ý}½™™±½…‘}­•åÌ¹…ÁÁ•¹¡½™™±½…‘}­•ä¤4(4(€€€€€€€€€€€¥˜¹½Ð¹•Ý}½™™±½…‘}­•åÌè4(€€€€€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹…‘Ù…¹•}ÍÑ½É•‘}¥‘à¡¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì¤4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(4(€€€€€€€€€€€ÍÑ½É•}½ÕÑÁÕÐ€ôÍ•±˜¹µ…¹…•È¹ÁÉ•Á…É•}ÍÑ½É” 4(€€€€€€€€€€€€€€€¹•Ý}½™™±½…‘}­•åÌ°É•Å}ÍÑ…ÑÕÌ¹É•Å}½¹Ñ•áÐ4(€€€€€€€€€€€€¤4(€€€€€€€€€€€¥˜ÍÑ½É•}½ÕÑÁÕÐ¥Ì9½¹”è4(€€€€€€€€€€€€€€€±½•È¹Ý…É¹¥¹œ ‰I•ÅÕ•ÍÐ€•Ìè…¹¹½ÐÍÑ½É”‰±½­Ìˆ°É•Å}¥¤4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(4(€€€€€€€€€€€¥˜¹½ÐÍÑ½É•}½ÕÑÁÕÐ¹­•åÍ}Ñ½}ÍÑ½É”è4(€€€€€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹…‘Ù…¹•}ÍÑ½É•‘}¥‘à¡¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì¤4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(4(€€€€€€€€€€€Í•±˜¹}Ñ½Õ ¡É•Å}ÍÑ…ÑÕÌ¤4(4(€€€€€€€€€€€­•åÍ}Ñ½}ÍÑ½É”€ôÍ•Ð¡ÍÑ½É•}½ÕÑÁÕÐ¹­•åÍ}Ñ½}ÍÑ½É”¤4(4(€€€€€€€€€€€É½ÕÁ}Í¥é•Ìè±¥ÍÑm¥¹Ñt€ômt4(€€€€€€€€€€€‰±½­}¥¹‘¥•Ìè±¥ÍÑm¥¹Ñt€ômt4(€€€€€€€€€€€ÍÉ}‰±½­}¥‘Ìè±¥ÍÑm¥¹Ñt€ômt4(€€€€€€€€€€€Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ìè±¥ÍÑm¥¹Ñt€ômt4(€€€€€€€€€€€¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ìè±¥ÍÑm¥¹Ñt€ômt4(€€€€€€€€€€€™½ÈÉ½ÕÁ}½¹™¥œ°É½ÕÁ}ÍÑ…Ñ”¥¸é¥À 4(€€€€€€€€€€€€€€€Í•±˜¹½¹™¥œ¹­Ù}É½ÕÁ}½¹™¥Ì°É•Å}ÍÑ…ÑÕÌ¹É½ÕÁ}ÍÑ…Ñ•Ì4(€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€¥Í}Í±¥‘¥¹}Ý¥¹‘½Ü€ô€ 4(€€€€€€€€€€€€€€€€€€€É½ÕÁ}½¹™¥œ¹Í±¥‘¥¹}Ý¥¹‘½Ý}Í¥é•}¥¹}‰±½­Ì¥Ì¹½Ð9½¹”4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€¹Õµ}‰±½­Ì€ô¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì€¼¼É½ÕÁ}½¹™¥œ¹½™™±½…‘•‘}‰±½­}Í¥é”4(€€€€€€€€€€€€€€€ÍÑ…ÉÑ}‰±½­}¥‘à€ôÉ½ÕÁ}ÍÑ…Ñ”¹¹•áÑ}ÍÑ½É•‘}‰±½­}¥‘à4(€€€€€€€€€€€€€€€‰±½­}¥‘Ì€ôÉ½ÕÁ}ÍÑ…Ñ”¹‰±½­}¥‘Ì4(€€€€€€€€€€€€€€€¹Õµ}É½ÕÁ}‰±½­Ì€ô€À4(€€€€€€€€€€€€€€€ÍÑ…ÉÑ}ÁÕ}‰±½­}¥‘àè¥¹Ðð9½¹”€ô9½¹”4(€€€€€€€€€€€€€€€™½È¥‘à°½™™±½…‘}­•ä¥¸•¹Õµ•É…Ñ” 4(€€€€€€€€€€€€€€€€€€€É½ÕÁ}ÍÑ…Ñ”¹½™™±½…‘}­•åÍmÍÑ…ÉÑ}‰±½­}¥‘àé¹Õµ}‰±½­Ít4(€€€€€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€€€€€¥˜½™™±½…‘}­•ä¹½Ð¥¸­•åÍ}Ñ½}ÍÑ½É”è4(€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(4(€€€€€€€€€€€€€€€€€€€½™™±½…‘•‘}‰±½­}¥‘à€ôÍÑ…ÉÑ}‰±½­}¥‘à€¬¥‘à4(€€€€€€€€€€€€€€€€€€€ÁÕ}‰±½­}¥‘à€ô½™™±½…‘•‘}‰±½­}¥‘à€¨‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€€€€€€€€€€€€€¹Õµ}É½ÕÁ}‰±½­Ì€¬ô‰±½­}Í¥é•}™…Ñ½È4(€€€€€€€€€€€€€€€€€€€™½È¤¥¸É…¹”¡‰±½­}Í¥é•}™…Ñ½È¤è4(€€€€€€€€€€€€€€€€€€€€€€€‰±½­}¥€ô‰±½­}¥‘ÍmÁÕ}‰±½­}¥‘à€¬¥t4(€€€€€€€€€€€€€€€€€€€€€€€¥˜‰±½­}¥€ôô€Àè4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€ŒÍ­¥ÁÁ•‰±½­Ì…¹¹½Ð…ÁÁ•…È…™Ñ•È¹½¸µÍ­¥ÁÁ•‰±½­Ì4(€€€€€€€€€€€€€€€€€€€€€€€€€€€…ÍÍ•ÉÐÍÑ…ÉÑ}ÁÕ}‰±½­}¥‘à¥Ì9½¹”4(€€€€€€€€€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€€€€€€€€€€€€€•±¥˜ÍÑ…ÉÑ}ÁÕ}‰±½­}¥‘à¥Ì9½¹”è4(€€€€€€€€€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ}ÁÕ}‰±½­}¥‘à€ôÁÕ}‰±½­}¥‘à€¬¤4(€€€€€€€€€€€€€€€€€€€€€€€ÍÉ}‰±½­}¥‘Ì¹…ÁÁ•¹¡‰±½­}¥¤4(€€€€€€€€€€€€€€€€€€€€€€€¥˜¥Í}Í±¥‘¥¹}Ý¥¹‘½Üè4(€€€€€€€€€€€€€€€€€€€€€€€€€€€Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì¹…ÁÁ•¹¡‰±½­}¥¤4(€€€€€€€€€€€€€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€€€€€€€€€€€€¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì¹…ÁÁ•¹¡‰±½­}¥¤4(4(€€€€€€€€€€€€€€€É½ÕÁ}Í¥é•Ì¹…ÁÁ•¹¡¹Õµ}É½ÕÁ}‰±½­Ì¤4(€€€€€€€€€€€€€€€‰±½­}¥¹‘¥•Ì¹…ÁÁ•¹¡ÍÑ…ÉÑ}ÁÕ}‰±½­}¥‘à½È€À¤4(€€€€€€€€€€€€€€€É½ÕÁ}ÍÑ…Ñ”¹¹•áÑ}ÍÑ½É•‘}‰±½­}¥‘à€ô¹Õµ}‰±½­Ì4(4(€€€€€€€€€€€ÍÉ}ÍÁ•Œ€ôAU1½…‘MÑ½É•MÁ•Œ 4(€€€€€€€€€€€€€€€ÍÉ}‰±½­}¥‘Ì°É½ÕÁ}Í¥é•ÌõÉ½ÕÁ}Í¥é•Ì°‰±½­}¥¹‘¥•Ìõ‰±½­}¥¹‘¥•Ì4(€€€€€€€€€€€€¤4(€€€€€€€€€€€‘ÍÑ}ÍÁ•Œ€ôÍÑ½É•}½ÕÑÁÕÐ¹ÍÑ½É•}ÍÁ•Œ4(4(€€€€€€€€€€€©½‰}¥€ôÍ•±˜¹}•¹•É…Ñ•}©½‰}¥ ¤4(€€€€€€€€€€€€Œ„ÍÑ½É”…¸½¹±ä‰”¥ÍÍÕ•Ý¡•¸¹¼±½…¥ÌÁ•¹‘¥¹œ¸4(€€€€€€€€€€€¥˜É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ìè4(€€€€€€€€€€€€€€€…¹å}©¥€ô¹•áÐ¡¥Ñ•È¡É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¤¤4(€€€€€€€€€€€€€€€…ÍÍ•ÉÐÍ•±˜¹}©½‰Ím…¹å}©¥‘t¹¥Í}ÍÑ½É”4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¹…‘¡©½‰}¥¤4(4(€€€€€€€€€€€€Œ]…Ñ Í±¥‘¥¹œÝ¥¹‘½Ü‰±½­Ì…ÌÑ¡•äµ…ä•Ð•Ù¥Ñ•4(€€€€€€€€€€€€Œ‰•™½É”Ñ¡”É•ÅÕ•ÍÐ™¥¹¥Í¡•Ì4(€€€€€€€€€€€™½È‰¥¥¸Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì½È€ ¤è4(€€€€€€€€€€€€€€€Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹Í•Ñ‘•™…Õ±Ð¡‰¥°Í•Ð ¤¤¹…‘¡©½‰}¥¤4(4(€€€€€€€€€€€€ŒÑ¡”¹½¸µÍ±¥‘¥¹œÝ¥¹‘½Ü‰±½­ÌÝ¥±°‰”Ý…Ñ¡•½¹±ä4(€€€€€€€€€€€€ŒÝ¡•¸Ñ¡”É•ÅÕ•ÍÐ™¥¹¥Í¡•Ì4(€€€€€€€€€€€Í•±˜¹}©½‰Ím©½‰}¥‘t€ôQÉ…¹Í™•É)½‰MÑ…ÑÕÌ 4(€€€€€€€€€€€€€€€É•Å}¥õÉ•Å}¥°4(€€€€€€€€€€€€€€€Á•¹‘¥¹}½Õ¹ÐõÍ•±˜¹½¹™¥œ¹¹Õµ}Ý½É­•ÉÌ°4(€€€€€€€€€€€€€€€­•åÌõÍ•Ð¡­•åÍ}Ñ½}ÍÑ½É”¤°4(€€€€€€€€€€€€€€€¥Í}ÍÑ½É”õQÉÕ”°4(€€€€€€€€€€€€€€€¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ìõ¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì°4(€€€€€€€€€€€€€€€Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘ÌõÍ±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì½È9½¹”°4(€€€€€€€€€€€€¤4(4(€€€€€€€€€€€ÍÑ½É•}©½‰Ím©½‰}¥‘t€ôQÉ…¹Í™•É)½ˆ 4(€€€€€€€€€€€€€€€É•Å}¥õÉ•Å}¥°ÑÉ…¹Í™•É}ÍÁ•Œô¡ÍÉ}ÍÁ•Œ°‘ÍÑ}ÍÁ•Œ¤4(€€€€€€€€€€€€¤4(4(€€€€€€€€€€€±½•È¹‘•‰Õœ 4(€€€€€€€€€€€€€€€€‰I•ÅÕ•ÍÐ€•Ì½™™±½…‘¥¹œ€•Ì‰±½­ÌÕÁÑ¼€•Ñ½­•¹Ì€¡©½ˆ€•¤ˆ°4(€€€€€€€€€€€€€€€É•Å}¥°4(€€€€€€€€€€€€€€€±•¸¡­•åÍ}Ñ½}ÍÑ½É”¤°4(€€€€€€€€€€€€€€€¹Õµ}½™™±½…‘…‰±•}Ñ½­•¹Ì°4(€€€€€€€€€€€€€€€©½‰}¥°4(€€€€€€€€€€€€¤4(4(€€€€€€€É•ÑÕÉ¸ÍÑ½É•}©½‰Ì4(4(€€€‘•˜‰Õ¥±‘}½¹¹•Ñ½É}µ•Ñ„ 4(€€€€€€€Í•±˜°Í¡•‘Õ±•É}½ÕÑÁÕÐèM¡•‘Õ±•É=ÕÑÁÕÐ4(€€€€¤€´ø-Y½¹¹•Ñ½É5•Ñ…‘…Ñ„è4(€€€€€€€™½ÈÉ•Å}¥¥¸Í¡•‘Õ±•É}½ÕÑÁÕÐ¹ÁÉ••µÁÑ•‘}É•Å}¥‘Ì½È€ ¤è4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ€ôÍ•±˜¹}É•Å}ÍÑ…ÑÕÌ¹•Ð¡É•Å}¥¤4(€€€€€€€€€€€¥˜É•Å}ÍÑ…ÑÕÌ¥Ì9½¹”½È¹½ÐÉ•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ìè4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€…¹å}©¥€ô¹•áÐ¡¥Ñ•È¡É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¤¤4(€€€€€€€€€€€…ÍÍ•ÉÐÍ•±˜¹}©½‰Ím…¹å}©¥‘t¹¥Í}ÍÑ½É”4(€€€€€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}©½‰Í}Ñ½}™±ÕÍ ¹ÕÁ‘…Ñ”¡É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¤4(4(€€€€€€€µ•Ñ„€ô=™™±½…‘¥¹½¹¹•Ñ½É5•Ñ…‘…Ñ„ 4(€€€€€€€€€€€±½…‘}©½‰ÌõÍ•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}±½…‘}©½‰Ì°4(€€€€€€€€€€€ÍÑ½É•}©½‰ÌõÍ•±˜¹}‰Õ¥±‘}ÍÑ½É•}©½‰Ì¡Í¡•‘Õ±•É}½ÕÑÁÕÐ¤°4(€€€€€€€€€€€©½‰Í}Ñ½}™±ÕÍ õÍ•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}©½‰Í}Ñ½}™±ÕÍ °4(€€€€€€€€¤4(€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}±½…‘}©½‰Ì€ôíô4(€€€€€€€Í•±˜¹}ÕÉÉ•¹Ñ}‰…Ñ¡}©½‰Í}Ñ½}™±ÕÍ €ôÍ•Ð ¤4(€€€€€€€É•ÑÕÉ¸µ•Ñ„4(4(€€€‘•˜ÕÁ‘…Ñ•}½¹¹•Ñ½É}½ÕÑÁÕÐ¡Í•±˜°½¹¹•Ñ½É}½ÕÑÁÕÐè-Y½¹¹•Ñ½É=ÕÑÁÕÐ¤è4(€€€€€€€€ˆˆˆ4(€€€€€€€UÁ‘…Ñ”-Y½¹¹•Ñ½ÈÍÑ…Ñ”™É½´Ý½É­•ÈµÍ¥‘”½¹¹•Ñ½ÉÌ½ÕÑÁÕÐ¸4(4(€€€€€€€ÉÌè4(€€€€€€€€€€€½¹¹•Ñ½É}½ÕÑÁÕÐ€¡-Y½¹¹•Ñ½É=ÕÑÁÕÐ¤èÑ¡”Ý½É­•ÈµÍ¥‘”4(€€€€€€€€€€€€€€€½¹¹•Ñ½ÉÌ½ÕÑÁÕÐ¸4(€€€€€€€€ˆˆˆ4(€€€€€€€µ•Ñ„€ô½¹¹•Ñ½É}½ÕÑÁÕÐ¹­Ù}½¹¹•Ñ½É}Ý½É­•É}µ•Ñ„4(€€€€€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡µ•Ñ„°=™™±½…‘¥¹]½É­•É5•Ñ…‘…Ñ„¤è4(€€€€€€€€€€€…ÍÍ•ÉÐµ•Ñ„¥Ì9½¹”4(€€€€€€€€€€€µ•Ñ„€ô=™™±½…‘¥¹]½É­•É5•Ñ…‘…Ñ„ ¤4(€€€€€€€™½È©½‰}¥°½Õ¹Ð¥¸µ•Ñ„¹½µÁ±•Ñ•‘}©½‰Ì¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€…ÍÍ•ÉÐ½Õ¹Ð€ø€À4(€€€€€€€€€€€©½‰}ÍÑ…ÑÕÌ€ôÍ•±˜¹}©½‰Ím©½‰}¥‘t4(€€€€€€€€€€€©½‰}ÍÑ…ÑÕÌ¹Á•¹‘¥¹}½Õ¹Ð€´ô½Õ¹Ð4(€€€€€€€€€€€¥˜©½‰}ÍÑ…ÑÕÌ¹Á•¹‘¥¹}½Õ¹Ð€ø€Àè4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€…ÍÍ•ÉÐ©½‰}ÍÑ…ÑÕÌ¹Á•¹‘¥¹}½Õ¹Ð€ôô€À4(4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ€ôÍ•±˜¹}É•Å}ÍÑ…ÑÕÍm©½‰}ÍÑ…ÑÕÌ¹É•Å}¥‘t4(€€€€€€€€€€€¥˜©½‰}ÍÑ…ÑÕÌ¹¥Í}ÍÑ½É”è4(€€€€€€€€€€€€€€€Í•±˜¹µ…¹…•È¹½µÁ±•Ñ•}ÍÑ½É”¡©½‰}ÍÑ…ÑÕÌ¹­•åÌ°É•Å}ÍÑ…ÑÕÌ¹É•Å}½¹Ñ•áÐ¤4(€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€Í•±˜¹µ…¹…•È¹½µÁ±•Ñ•}±½…¡©½‰}ÍÑ…ÑÕÌ¹­•åÌ°É•Å}ÍÑ…ÑÕÌ¹É•Å}½¹Ñ•áÐ¤4(€€€€€€€€€€€€€€€¥˜Í•±˜¹}‰±½­Í}‰•¥¹}±½…‘•è4(€€€€€€€€€€€€€€€€€€€Í•±˜¹}‰±½­Í}‰•¥¹}±½…‘•¹‘¥™™•É•¹•}ÕÁ‘…Ñ”¡©½‰}ÍÑ…ÑÕÌ¹­•åÌ¤4(€€€€€€€€€€€¥˜Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ìè4(€€€€€€€€€€€€€€€€ŒM±¥‘¥¹œÝ¥¹‘½Ü‰±½­Ì…É”ÑÉ…­•™É½´ÍÑ½É”É•…Ñ¥½¸4(€€€€€€€€€€€€€€€€Œ…¹µÕÍÐ‰”±•…¹•ÕÀÕ¹½¹‘¥Ñ¥½¹…±±ä¸4(€€€€€€€€€€€€€€€Í•±˜¹}É•µ½Ù•}Á•¹‘¥¹}©½ˆ¡©½‰}¥°©½‰}ÍÑ…ÑÕÌ¹Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì¤4(€€€€€€€€€€€€€€€€Œ9½¸µÍ±¥‘¥¹œµÝ¥¹‘½Ü‰±½­Ì…É”½¹±äÑÉ…­•…™Ñ•È4(€€€€€€€€€€€€€€€€ŒÉ•ÅÕ•ÍÑ}™¥¹¥Í¡•°Í¼½¹±ä±•…¸ÕÀ™½È™¥¹¥Í¡•É•ÅÕ•ÍÑÌ¸4(€€€€€€€€€€€€€€€¥˜É•Å}ÍÑ…ÑÕÌ¹É•Ä¹¥Í}™¥¹¥Í¡• ¤è4(€€€€€€€€€€€€€€€€€€€Í•±˜¹}É•µ½Ù•}Á•¹‘¥¹}©½ˆ 4(€€€€€€€€€€€€€€€€€€€€€€€©½‰}¥°©½‰}ÍÑ…ÑÕÌ¹¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì4(€€€€€€€€€€€€€€€€€€€€¤4(4(€€€€€€€€€€€‘•°Í•±˜¹}©½‰Ím©½‰}¥‘t4(€€€€€€€€€€€É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì¹É•µ½Ù”¡©½‰}¥¤4(€€€€€€€€€€€¥˜¹½ÐÉ•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ì…¹É•Å}ÍÑ…ÑÕÌ¹É•Ä¹¥Í}™¥¹¥Í¡• ¤è4(€€€€€€€€€€€€€€€‘•°Í•±˜¹}É•Å}ÍÑ…ÑÕÍm©½‰}ÍÑ…ÑÕÌ¹É•Å}¥‘t4(4(€€€‘•˜É•ÅÕ•ÍÑ}™¥¹¥Í¡• 4(€€€€€€€Í•±˜°4(€€€€€€€É•ÅÕ•ÍÐèI•ÅÕ•ÍÐ°4(€€€€¤€´øÑÕÁ±•m‰½½°°‘¥ÑmÍÑÈ°¹åtð9½¹•tè4(€€€€€€€€ˆˆˆ4(€€€€€€€…±±•Ý¡•¸„É•ÅÕ•ÍÐ¡…Ì™¥¹¥Í¡•°‰•™½É”¥ÑÌ‰±½­Ì…É”™É••¸4(4(€€€€€€€I•ÑÕÉ¹Ìè4(€€€€€€€€€€€QÉÕ”¥˜Ñ¡”É•ÅÕ•ÍÐ¥Ì‰•¥¹œÍ…Ù•½Í•¹Ð…Íå¹¡É½¹½ÕÍ±ä…¹‰±½­Ì4(€€€€€€€€€€€Í¡½Õ±¹½Ð‰”™É••Õ¹Ñ¥°Ñ¡”É•ÅÕ•ÍÑ}¥¥ÌÉ•ÑÕÉ¹•™É½´4(€€€€€€€€€€€•Ñ}™¥¹¥Í¡• ¤¸4(€€€€€€€€€€€=ÁÑ¥½¹…°-YQÉ…¹Í™•ÉA…É…µÌÑ¼‰”¥¹±Õ‘•¥¸Ñ¡”É•ÅÕ•ÍÐ½ÕÑÁÕÑÌ4(€€€€€€€€€€€É•ÑÕÉ¹•‰äÑ¡”•¹¥¹”¸4(€€€€€€€€ˆˆˆ4(€€€€€€€€ŒQ=<¡½É½é•Éä¤èÁ½ÍÍ¥‰±ä­¥­½™˜½™™±½…™½È±…ÍÐ‰±½¬4(€€€€€€€€ŒÝ¡¥ µ…ä¡…Ù”‰••¸‘•™•ÉÉ•‘Õ”Ñ¼…Íå¹ŒÍ¡•‘Õ±¥¹œ4(€€€€€€€É•Å}ÍÑ…ÑÕÌ€ôÍ•±˜¹}É•Å}ÍÑ…ÑÕÌ¹•Ð¡É•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥¤4(€€€€€€€¥˜É•Å}ÍÑ…ÑÕÌ¥Ì9½¹”è4(€€€€€€€€€€€É•ÑÕÉ¸…±Í”°9½¹”4(€€€€€€€¥˜¹½ÐÉ•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ìè4(€€€€€€€€€€€‘•°Í•±˜¹}É•Å}ÍÑ…ÑÕÍmÉ•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥‘t4(€€€€€€€€€€€É•ÑÕÉ¸…±Í”°9½¹”4(€€€€€€€€ŒA•¹‘¥¹œÍÑ½É•ÌÝ¥±°½ÕÑ±¥Ù”Ñ¡”É•ÅÕ•ÍÐÌ‰±½¬½Ý¹•ÉÍ¡¥À¸4(€€€€€€€€ŒI•¥ÍÑ•ÈÑ¡•´Í¼™ÕÑÕÉ”‰±½¬É•ÕÍ”ÑÉ¥•ÉÌ„™±ÕÍ ¸4(€€€€€€€™½È©½‰}¥¥¸É•Å}ÍÑ…ÑÕÌ¹ÑÉ…¹Í™•É}©½‰Ìè4(€€€€€€€€€€€©½‰}ÍÑ…ÑÕÌ€ôÍ•±˜¹}©½‰Ím©½‰}¥‘t4(€€€€€€€€€€€™½È‰¥¥¸©½‰}ÍÑ…ÑÕÌ¹¹½¹}Í±¥‘¥¹}Ý¥¹‘½Ý}‰±½­}¥‘Ì½È€ ¤è4(€€€€€€€€€€€€€€€Í•±˜¹}‰±½­}¥‘}Ñ½}Á•¹‘¥¹}©½‰Ì¹Í•Ñ‘•™…Õ±Ð¡‰¥°Í•Ð ¤¤¹…‘¡©½‰}¥¤4(€€€€€€€É•ÑÕÉ¸…±Í”°9½¹”4(4(€€€‘•˜Ñ…­•}•Ù•¹ÑÌ¡Í•±˜¤€´ø%Ñ•É…‰±•m-Y…¡•Ù•¹Ñtè4(€€€€€€€€ˆˆ‰Q…­”Ñ¡”-X…¡”•Ù•¹ÑÌ™É½´Ñ¡”½¹¹•Ñ½È¸4(4(€€€€€€€I•ÑÕÉ¹Ìè4(€€€€€€€€€€€±¥ÍÐ½˜-X…¡”•Ù•¹ÑÌ¸4(€€€€€€€€ˆˆˆ4(€€€€€€€™½È•Ù•¹Ð¥¸Í•±˜¹µ…¹…•È¹Ñ…­•}•Ù•¹ÑÌ ¤è4(€€€€€€€€€€€‰±½­}¡…Í¡•Ì€ôm•Ñ}½™™±½…‘}‰±½­}¡…Í ¡­•ä¤™½È­•ä¥¸•Ù•¹Ð¹­•åÍt4(€€€€€€€€€€€¥˜•Ù•¹Ð¹É•µ½Ù•è4(€€€€€€€€€€€€€€€å¥•±	±½­I•µ½Ù•¡‰±½­}¡…Í¡•Ìõ‰±½­}¡…Í¡•Ì°µ•‘¥Õ´õ•Ù•¹Ð¹µ•‘¥Õ´¤4(€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€å¥•±	±½­MÑ½É• 4(€€€€€€€€€€€€€€€€€€€‰±½­}¡…Í¡•Ìõ‰±½­}¡…Í¡•Ì°4(€€€€€€€€€€€€€€€€€€€Á…É•¹Ñ}‰±½­}¡…Í õ9½¹”°4(€€€€€€€€€€€€€€€€€€€Ñ½­•¹}¥‘Ìõmt°4(€€€€€€€€€€€€€€€€€€€±½É…}¥õ9½¹”°4(€€€€€€€€€€€€€€€€€€€‰±½­}Í¥é”ôÀ°4(€€€€€€€€€€€€€€€€€€€µ•‘¥Õ´õ•Ù•¹Ð¹µ•‘¥Õ´°4(€€€€€€€€€€€€€€€€€€€±½É…}¹…µ”õ9½¹”°4(€€€€€€€€€€€€€€€€¤4(4(€€€‘•˜Í¡ÕÑ‘½Ý¸¡Í•±˜¤€´ø9½¹”è4(€€€€€€€Í•±˜¹µ…¹…•È¹Í¡ÕÑ‘½Ý¸ ¤4(