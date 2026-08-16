# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    TQFullAttentionSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add locally cached blocks to a new request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        # The coordinator handles the running-request fast path before entering
        # either phase, so every cache group sees the same request state.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Allocate connector-hit blocks after all groups touched local hits."""
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        req_blocks.extend(allocated_blocks)
        if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec):
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if type(self.kv_cache_spec) in (FullAttentionSpec, TQFullAttentionSpec):
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignmeï¾|¶‰žËkºwµç}µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð¤€´ø9½¹”è4(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í•±˜¹­Ù}…¡•}ÍÁ•Œ°5…µ‰…MÁ•Œ¤4(4(€€€€€€€€Œ9=Q€¡Ñ‘½Õ‰±•À¤Ý¥Ñ …Íå¹ŒÍ¡•‘Õ±¥¹œ°Ñ¡”¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì…¸½¹Ñ…¥¸4(€€€€€€€€Œ‘É…™ÐÑ½­•¹Ì™É½´Ñ¡”ÁÉ•Ù¥½ÕÌÍÑ•ÀÑ¡…Ðµ…ä½Èµ…ä¹½Ð‰”É•©•Ñ•±…Ñ•È¸4(€€€€€€€€ŒQ¡¥Ì…¸µ…­”ÕÌÑ¡¥¹¬Ý”…É”™ÕÉÑ¡•È…¡•…¥¸Ñ¡”Í•ÅÕ•¹”Ñ¡…¸Ý”…ÑÕ…±±ä4(€€€€€€€€Œ…É”°Í¼±•ÐÌ…ÍÍÕµ”Ñ¡…Ð…±°Ñ½­•¹Ì…É”É•©•Ñ•Í¼Ý”‘½¸Ð™É•”‰±½­Ì4(€€€€€€€€ŒÑ¡…ÐÝ”µ¥¡Ð…ÑÕ…±±ä¹••¸4(€€€€€€€¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì€ôµ…à À°¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì€´Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì¤4(4(€€€€€€€ÍÕÁ•È ¤¹É•µ½Ù•}Í­¥ÁÁ•‘}‰±½­Ì¡É•ÅÕ•ÍÑ}¥°¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì¤4(€€€€€€€¥˜Í•±˜¹µ…µ‰…}…¡•}µ½‘”€ôô€‰…±¥¸ˆè4(€€€€€€€€€€€€Œ±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘á€É•™•ÉÌÑ¼Ñ¡”‰±½¬¥¹‘•à…±±½…Ñ•ÑÝ¼ÍÑ•ÁÌ…¼¸4(€€€€€€€€€€€€ŒQ¡”‰±½¬…±±½…Ñ•¥¸Ñ¡”ÁÉ•Ù¥½ÕÌÍÑ•À¥ÌÕÍ•Ñ¼½Áä5…µ‰„ÍÑ…Ñ•Ì4(€€€€€€€€€€€€Œ¥¹Ñ¼Ñ¡”‰±½¬…±±½…Ñ•¥¸Ñ¡”ÕÉÉ•¹ÐÍÑ•ÀìÑ¡”•…É±¥•È‰±½¬¥Ì4(€€€€€€€€€€€€Œ¹¼±½¹•È¹••‘•…¹Í¡½Õ±‰”™É••¡•É”¸4(€€€€€€€€€€€±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘à€ôÍ•±˜¹±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘à¹•Ð¡É•ÅÕ•ÍÑ}¥¤4(€€€€€€€€€€€€Œ	±½­Ì…±±½…Ñ•‘ÕÉ¥¹œÁÉ•™¥±°µ…ä‰”¹½¸µ½¹Ñ¥Õ½ÕÌ¸UÍ”4(€€€€€€€€€€€€Œ±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘á€Ñ¼™É•”Ñ¡”…ÁÁÉ½ÁÉ¥…Ñ”‰±½¬…¹É•Á±…”¥Ð4(€€€€€€€€€€€€ŒÝ¥Ñ „¹Õ±°‰±½¬¸4(€€€€€€€€€€€¥˜€ 4(€€€€€€€€€€€€€€€±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘à¥Ì¹½Ð9½¹”4(€€€€€€€€€€€€€€€…¹±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘à4(€€€€€€€€€€€€€€€€ð‘¥Ø¡¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì°Í•±˜¹‰±½­}Í¥é”¤€´€Ä4(€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€‰±½­Ì€ôÍ•±˜¹É•Å}Ñ½}‰±½­ÍmÉ•ÅÕ•ÍÑ}¥‘t4(€€€€€€€€€€€€€€€¥˜‰±½­Ím±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘át€„ôÍ•±˜¹}¹Õ±±}‰±½¬è4(€€€€€€€€€€€€€€€€€€€Í•±˜¹‰±½­}Á½½°¹™É••}‰±½­Ì¡m‰±½­Ím±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘áut¤4(€€€€€€€€€€€€€€€€€€€‰±½­Ím±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘át€ôÍ•±˜¹}¹Õ±±}‰±½¬4(4(€€€‘•˜•Ñ}¹Õµ}½µµ½¹}ÁÉ•™¥á}‰±½­Ì¡Í•±˜°ÉÕ¹¹¥¹}É•ÅÕ•ÍÑ}¥èÍÑÈ¤€´ø¥¹Ðè4(€€€€€€€€ˆˆˆ4(€€€€€€€…Í…‘”…ÑÑ•¹Ñ¥½¸¥Ì¹½ÐÍÕÁÁ½ÉÑ•‰äµ…µ‰„4(€€€€€€€€ˆˆˆ4(€€€€€€€É•ÑÕÉ¸€À4(4(€€€‘•˜•Ñ}¹Õµ}‰±½­Í}Ñ½}…±±½…Ñ” 4(€€€€€€€Í•±˜°4(€€€€€€€É•ÅÕ•ÍÑ}¥èÍÑÈ°4(€€€€€€€¹Õµ}Ñ½­•¹Ìè¥¹Ð°4(€€€€€€€¹•Ý}½µÁÕÑ•‘}‰±½­ÌèM•ÅÕ•¹•m-Y…¡•	±½­t°4(€€€€€€€Ñ½Ñ…±}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€€€€€¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°è¥¹Ð°4(€€€€€€€…ÁÁ±å}…‘µ¥ÍÍ¥½¹}…Àè‰½½°€ô…±Í”°4(€€€€¤€´ø¥¹Ðè4(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í•±˜¹­Ù}…¡•}ÍÁ•Œ°5…µ‰…MÁ•Œ¤4(€€€€€€€¥˜€ 4(€€€€€€€€€€€±•¸¡¹•Ý}½µÁÕÑ•‘}‰±½­Ì¤€ø€À4(€€€€€€€€€€€…¹¹•Ý}½µÁÕÑ•‘}‰±½­Íl´Åt¹‰±½­}¡…Í ¥¸Í•±˜¹…¡•‘}‰±½­Í}Ñ¡¥Í}ÍÑ•À4(€€€€€€€€¤è4(€€€€€€€€€€€€Œ5…µ‰„…¸ÐÉ•±ä½¸‰±½­Ì•¹•É…Ñ•‰ä½Ñ¡•ÈÉ•ÅÕ•ÍÑÌ¥¸Ñ¡”ÕÉÉ•¹ÐÍÑ•À4(€€€€€€€€€€€€ŒQ¼ÁÕÐ¥Ð¥¸Ñ¡”¹•áÐÍÑ•À°Ý”É•ÑÕÉ¸¹Õµ}ÁÕ}‰±½­Ì€¬€ÄÍ¼4(€€€€€€€€€€€€ŒÑ¡…Ð­Ù}…¡•}µ…¹…•ÈÝ¥±°Ñ¡¥¹¬Ñ¡•É”¥Ì¹¼•¹½Õ ‰±½­ÌÑ¼…±±½…Ñ”¹½Ü4(€€€€€€€€€€€€Œ…¹‘½¸ÐÍ¡•‘Õ±”¥Ð¥¸Ñ¡”ÕÉÉ•¹ÐÍÑ•À¸4(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜¹‰±½­}Á½½°¹¹Õµ}ÁÕ}‰±½­Ì€¬€Ä4(€€€€€€€¥˜Í•±˜¹µ…µ‰…}…¡•}µ½‘”€„ô€‰…±¥¸ˆè4(€€€€€€€€€€€€Œ±±½…Ñ”•áÑÉ„¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Í€‰±½­Ì™½È4(€€€€€€€€€€€€ŒÍÁ•Õ±…Ñ¥Ù”‘•½‘¥¹œ€¡5Q@½1¤Ý¥Ñ ±¥¹•…È…ÑÑ•¹Ñ¥½¸¸4(€€€€€€€€€€€¥˜Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì€ø€Àè4(€€€€€€€€€€€€€€€¹Õµ}Ñ½­•¹Ì€¬ô€ 4(€€€€€€€€€€€€€€€€€€€Í•±˜¹­Ù}…¡•}ÍÁ•Œ¹‰±½­}Í¥é”€¨Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€É•ÑÕÉ¸ÍÕÁ•È ¤¹•Ñ}¹Õµ}‰±½­Í}Ñ½}…±±½…Ñ” 4(€€€€€€€€€€€€€€€É•ÅÕ•ÍÑ}¥°4(€€€€€€€€€€€€€€€¹Õµ}Ñ½­•¹Ì°4(€€€€€€€€€€€€€€€¹•Ý}½µÁÕÑ•‘}‰±½­Ì°4(€€€€€€€€€€€€€€€Ñ½Ñ…±}½µÁÕÑ•‘}Ñ½­•¹Ì°4(€€€€€€€€€€€€€€€¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°°4(€€€€€€€€€€€€€€€…ÁÁ±å}…‘µ¥ÍÍ¥½¹}…Àõ…ÁÁ±å}…‘µ¥ÍÍ¥½¹}…À°4(€€€€€€€€€€€€¤4(€€€€€€€•±Í”è4(€€€€€€€€€€€€Œ]”‘½¸Ð…±±½…Ñ”‰±½­Ì™½È±½½­…¡•…Ñ½­•¹Ì¥¸…±¥¸µ½‘”°‰•…ÕÍ”¥˜4(€€€€€€€€€€€€Œà€¨‰±½­}Í¥é”Ñ½­•¹Ì…É”Í¡•‘Õ±•°¹Õµ}Ñ½­•¹Ì¥Ì4(€€€€€€€€€€€€Œà€¨‰±½­}Í¥é”€¬¹Õµ}±½½­…¡•…‘}Ñ½­•¹Ì…¹‰É•…­ÌÑ¡”…±¥¹µ•¹Ð¸4(€€€€€€€€€€€€Œ]”…¸¥¹½É”±½½­…¡•…Ñ½­•¹Ì‰•…ÕÍ”ÕÉÉ•¹Ð‘É…™Ðµ½‘•±Ì‘½¸Ð¡…Ù”4(€€€€€€€€€€€€Œµ…µ‰„±…å•ÉÌ¸4(€€€€€€€€€€€¹Õµ}Ñ½­•¹Ì€ô¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°4(4(€€€€€€€€€€€€Œ9=Q¡Ñ‘½Õ‰±”¤èÑ¡¥Ì¥Ì…¸½Ù•Èµ•ÍÑ¥µ…Ñ”½˜¡½Üµ…¹ä‰±½­ÌÝ”¹••‰•…ÕÍ”4(€€€€€€€€€€€€Œ¹Õµ}Ñ½­•¹Ì…¸¥¹±Õ‘”‘É…™ÐÑ½­•¹ÌÑ¡…ÐÝ¥±°±…Ñ•È‰”É•©•Ñ•¸4(€€€€€€€€€€€¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€ô€ 4(€€€€€€€€€€€€€€€‘¥Ø¡¹Õµ}Ñ½­•¹Ì°Í•±˜¹‰±½­}Í¥é”¤€¬Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(€€€€€€€€€€€€¤4(€€€€€€€€€€€¹Õµ}¹•Ý}‰±½­Ì€ô€ 4(€€€€€€€€€€€€€€€¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì4(€€€€€€€€€€€€€€€€´±•¸¡¹•Ý}½µÁÕÑ•‘}‰±½­Ì¤4(€€€€€€€€€€€€€€€€´±•¸¡Í•±˜¹É•Å}Ñ½}‰±½­ÍmÉ•ÅÕ•ÍÑ}¥‘t¤4(€€€€€€€€€€€€¤4(€€€€€€€€€€€¥˜¹Õµ}¹•Ý}‰±½­Ì€ø€Àè4(€€€€€€€€€€€€€€€¥˜É•ÅÕ•ÍÑ}¥¥¸Í•±˜¹}…±±½…Ñ•‘}‰±½­}É•ÅÌè4(€€€€€€€€€€€€€€€€€€€€Œ=±É•ÅÕ•ÍÐ¸9••‘Ì…Ðµ½ÍÐ€Äµ½É”‰±½­Ì…ÌÝ”…¸É•ÕÍ”Ñ¡”4(€€€€€€€€€€€€€€€€€€€€ŒÍÁ•Õ±…Ñ¥Ù”‰±½­Ì¥¸ÁÉ•Ù¥½ÕÌÍÑ•À¸4(€€€€€€€€€€€€€€€€€€€¹Õµ}¹•Ý}‰±½­Ì€ô€Ä4(€€€€€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€€€€€Œ¥ÉÍÐÁÉ•™¥±°¸±±½…Ñ”€Ä‰±½¬™½ÈÉÕ¹¹¥¹œÍÑ…Ñ”…¹Ñ¡”4(€€€€€€€€€€€€€€€€€€€€ŒÍÁ•Õ±…Ñ¥Ù”‰±½­Ì¸4(€€€€€€€€€€€€€€€€€€€¹Õµ}¹•Ý}‰±½­Ì€ô€Ä€¬Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(4(€€€€€€€€€€€¹Õµ}•Ù¥Ñ…‰±•}½µÁÕÑ•‘}‰±½­Ì€ôÍ•±˜¹}•Ñ}¹Õµ}•Ù¥Ñ…‰±•}‰±½­Ì 4(€€€€€€€€€€€€€€€¹•Ý}½µÁÕÑ•‘}‰±½­Ì4(€€€€€€€€€€€€¤4(€€€€€€€€€€€É•ÑÕÉ¸¹Õµ}¹•Ý}‰±½­Ì€¬¹Õµ}•Ù¥Ñ…‰±•}½µÁÕÑ•‘}‰±½­Ì4(4(€€€‘•˜…±±½…Ñ•}¹•Ý}‰±½­Ì 4(€€€€€€€Í•±˜°É•ÅÕ•ÍÑ}¥èÍÑÈ°¹Õµ}Ñ½­•¹Ìè¥¹Ð°¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°è¥¹Ð4(€€€€¤€´ø±¥ÍÑm-Y…¡•	±½­tè4(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡Í•±˜¹­Ù}…¡•}ÍÁ•Œ°5…µ‰…MÁ•Œ¤4(€€€€€€€¥˜Í•±˜¹µ…µ‰…}…¡•}µ½‘”€„ô€‰…±¥¸ˆè4(€€€€€€€€€€€€Œ±±½…Ñ”•áÑÉ„¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Í€‰±½­Ì™½È4(€€€€€€€€€€€€ŒÍÁ•Õ±…Ñ¥Ù”‘•½‘¥¹œ€¡5Q@½1¤Ý¥Ñ ±¥¹•…È…ÑÑ•¹Ñ¥½¸¸4(€€€€€€€€€€€¥˜Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì€ø€Àè4(€€€€€€€€€€€€€€€¹Õµ}Ñ½­•¹Ì€¬ôÍ•±˜¹‰±½­}Í¥é”€¨Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(€€€€€€€€€€€É•ÑÕÉ¸ÍÕÁ•È ¤¹…±±½…Ñ•}¹•Ý}‰±½­Ì 4(€€€€€€€€€€€€€€€É•ÅÕ•ÍÑ}¥°¹Õµ}Ñ½­•¹Ì°¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°4(€€€€€€€€€€€€¤4(€€€€€€€•±Í”è4(€€€€€€€€€€€€Œ]”‘½¸Ð…±±½…Ñ”‰±½­Ì™½È±½½­…¡•…Ñ½­•¹Ì¥¸…±¥¸µ½‘”°‰•…ÕÍ”¥˜4(€€€€€€€€€€€€Œà€¨‰±½­}Í¥é”Ñ½­•¹Ì…É”Í¡•‘Õ±•°¹Õµ}Ñ½­•¹Ì¥Ì4(€€€€€€€€€€€€Œà€¨‰±½­}Í¥é”€¬¹Õµ}±½½­…¡•…‘}Ñ½­•¹Ì…¹‰É•…­ÌÑ¡”…±¥¹µ•¹Ð¸4(€€€€€€€€€€€€Œ]”…¸¥¹½É”±½½­…¡•…Ñ½­•¹Ì‰•…ÕÍ”ÕÉÉ•¹Ð‘É…™Ðµ½‘•±Ì‘½¸Ð¡…Ù”4(€€€€€€€€€€€€Œµ…µ‰„±…å•ÉÌ¸4(€€€€€€€€€€€¹Õµ}Ñ½­•¹Ì€ô¹Õµ}Ñ½­•¹Í}µ…¥¹}µ½‘•°4(€€€€€€€€€€€É•Å}‰±½­Ìè±¥ÍÑm-Y…¡•	±½­t€ôÍ•±˜¹É•Å}Ñ½}‰±½­ÍmÉ•ÅÕ•ÍÑ}¥‘t4(€€€€€€€€€€€€Œ9=Q¡Ñ‘½Õ‰±”¤èÑ¡¥Ì¥Ì…¸½Ù•Èµ•ÍÑ¥µ…Ñ”½˜¡½Üµ…¹ä‰±½­ÌÝ”¹••‰•…ÕÍ”4(€€€€€€€€€€€€Œ¹Õµ}Ñ½­•¹Ì…¸¥¹±Õ‘”‘É…™ÐÑ½­•¹ÌÑ¡…ÐÝ¥±°±…Ñ•È‰”É•©•Ñ•¸4(€€€€€€€€€€€¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€ô€ 4(€€€€€€€€€€€€€€€‘¥Ø¡¹Õµ}Ñ½­•¹Ì°Í•±˜¹‰±½­}Í¥é”¤€¬Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(€€€€€€€€€€€€¤4(€€€€€€€€€€€¥˜¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€ôô±•¸¡É•Å}‰±½­Ì¤è4(€€€€€€€€€€€€€€€É•ÑÕÉ¸mt4(€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€…ÍÍ•ÉÐ¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€ø±•¸¡É•Å}‰±½­Ì¤°€ 4(€€€€€€€€€€€€€€€€€€€€‰¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€ˆ4(€€€€€€€€€€€€€€€€€€€˜‰í¹Õµ}É•ÅÕ¥É•‘}‰±½­Íô€ð±•¸¡É•Å}‰±½­Ì¤í±•¸¡É•Å}‰±½­Ì¥ôˆ4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€ÁÉ•Ù}‰±½­}±•¸€ô±•¸¡É•Å}‰±½­Ì¤4(€€€€€€€€€€€€€€€‰±½­Í}…±±½…Ñ•€ôÉ•ÅÕ•ÍÑ}¥¥¸Í•±˜¹}…±±½…Ñ•‘}‰±½­}É•ÅÌ4(€€€€€€€€€€€€€€€€ŒI•½ÉÑ¡”±…ÍÐÍÑ…Ñ”‰±½¬4(€€€€€€€€€€€€€€€¥˜‰±½­Í}…±±½…Ñ•è4(€€€€€€€€€€€€€€€€€€€€Œ]”…±Ý…åÌÍ…Ù”Ñ¡”ÉÕ¹¹¥¹œÍÑ…Ñ”…ÐÑ¡”±…ÍÐ4(€€€€€€€€€€€€€€€€€€€€Œ€ Ä€¬¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì¤‰±½¬4(€€€€€€€€€€€€€€€€€€€Í•±˜¹±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘ámÉ•ÅÕ•ÍÑ}¥‘t€ô€ 4(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ•Ù}‰±½­}±•¸€´€Ä€´Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì4(€€€€€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€•±¥˜ÁÉ•Ù}‰±½­}±•¸€ø€Àè4(€€€€€€€€€€€€€€€€€€€€Œ]¡•¸„¹•ÜÉ•ÅÕ•ÍÐ¡¥ÑÌÑ¡”ÁÉ•™¥à…¡”°Ñ¡”±…ÍÐ‰±½¬4(€€€€€€€€€€€€€€€€€€€€ŒÍ…Ù•ÌÑ¡”¡¥ÐÍÑ…Ñ”¸4(€€€€€€€€€€€€€€€€€€€Í•±˜¹±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘ámÉ•ÅÕ•ÍÑ}¥‘t€ôÁÉ•Ù}‰±½­}±•¸€´€Ä4(4(€€€€€€€€€€€€€€€¹Õµ}Í­¥ÁÁ•‘}‰±½­Ì€ô€ 4(€€€€€€€€€€€€€€€€€€€¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€´Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì€´€Ä4(€€€€€€€€€€€€€€€€¤4(€€€€€€€€€€€€€€€€Œ¹Õ±°‰±½­Ì4(€€€€€€€€€€€€€€€¥˜ÁÉ•Ù}‰±½­}±•¸€ð¹Õµ}Í­¥ÁÁ•‘}‰±½­Ìè4(€€€€€€€€€€€€€€€€€€€É•Å}‰±½­Ì¹•áÑ•¹ 4(€€€€€€€€€€€€€€€€€€€€€€€l4(€€€€€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹}¹Õ±±}‰±½¬4(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½È|¥¸É…¹”¡ÁÉ•Ù}‰±½­}±•¸°¹Õµ}Í­¥ÁÁ•‘}‰±½­Ì¤4(€€€€€€€€€€€€€€€€€€€€€€€t4(€€€€€€€€€€€€€€€€€€€€¤4(4(€€€€€€€€€€€€€€€¥˜‰±½­Í}…±±½…Ñ•è4(€€€€€€€€€€€€€€€€€€€€ŒÉ•ÕÍ”ÁÉ•Ù¥½ÕÌÍÁ•Õ±…Ñ¥Ù”‰±½­Ì¥¸Ñ¡¥ÌÍÑ•À4(€€€€€€€€€€€€€€€€€€€™½È‰±½­}¥‘à¥¸É…¹” 4(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ•Ù}‰±½­}±•¸€´Í•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì°ÁÉ•Ù}‰±½­}±•¸4(€€€€€€€€€€€€€€€€€€€€¤è4(€€€€€€€€€€€€€€€€€€€€€€€¥˜‰±½­}¥‘à€ð¹Õµ}Í­¥ÁÁ•‘}‰±½­Ìè4(€€€€€€€€€€€€€€€€€€€€€€€€€€€É•Å}‰±½­Ì¹…ÁÁ•¹¡É•Å}‰±½­Ím‰±½­}¥‘át¤4(€€€€€€€€€€€€€€€€€€€€€€€€€€€É•Å}‰±½­Ím‰±½­}¥‘át€ôÍ•±˜¹}¹Õ±±}‰±½¬4(€€€€€€€€€€€€€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€€€€€€€€€€€€‰É•…¬4(€€€€€€€€€€€€€€€¹Õµ}¹•Ý}‰±½­Ì€ô¹Õµ}É•ÅÕ¥É•‘}‰±½­Ì€´±•¸¡É•Å}‰±½­Ì¤4(€€€€€€€€€€€€€€€¥˜‰±½­Í}…±±½…Ñ•è4(€€€€€€€€€€€€€€€€€€€…ÍÍ•ÉÐ¹Õµ}¹•Ý}‰±½­Ì€ðô€Ä4(€€€€€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€€€€…ÍÍ•ÉÐ¹Õµ}¹•Ý}‰±½­Ì€ðôÍ•±˜¹¹Õµ}ÍÁ•Õ±…Ñ¥Ù•}‰±½­Ì€¬€Ä4(€€€€€€€€€€€€€€€¹•Ý}‰±½­Ì€ôÍ•±˜¹‰±½­}Á½½°¹•Ñ}¹•Ý}‰±½­Ì¡¹Õµ}¹•Ý}‰±½­Ì¤4(€€€€€€€€€€€€€€€É•Å}‰±½­Ì¹•áÑ•¹¡¹•Ý}‰±½­Ì¤4(€€€€€€€€€€€€€€€Í•±˜¹}…±±½…Ñ•‘}‰±½­}É•ÅÌ¹…‘¡É•ÅÕ•ÍÑ}¥¤4(€€€€€€€€€€€€€€€É•ÑÕÉ¸É•Å}‰±½­ÍmÁÉ•Ù}‰±½­}±•¸ét4(4(€€€‘•˜™É•”¡Í•±˜°É•ÅÕ•ÍÑ}¥èÍÑÈ¤€´ø9½¹”è4(€€€€€€€¥˜Í•±˜¹µ…µ‰…}…¡•}µ½‘”€ôô€‰…±¥¸ˆè4(€€€€€€€€€€€Í•±˜¹}…±±½…Ñ•‘}‰±½­}É•ÅÌ¹‘¥Í…É¡É•ÅÕ•ÍÑ}¥¤4(€€€€€€€€€€€Í•±˜¹±…ÍÑ}ÍÑ…Ñ•}‰±½­}¥‘à¹Á½À¡É•ÅÕ•ÍÑ}¥°9½¹”¤4(€€€€€€€ÍÕÁ•È ¤¹™É•”¡É•ÅÕ•ÍÑ}¥¤4(4(€€€‘•˜•Ñ}¹Õµ}Í­¥ÁÁ•‘}Ñ½­•¹Ì¡Í•±˜°¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð¤€´ø¥¹Ðè4(€€€€€€€€ˆˆˆ4(€€€€€€€•ÐÑ¡”¹Õµ‰•È½˜Ñ½­•¹ÌÝ¡½Í”µ…µ‰„ÍÑ…Ñ”…É”¹½Ð¹••‘•…¹åµ½É”¸5…µ‰„½¹±ä4(€€€€€€€¹••Ñ¼­••ÀÑ¡”ÍÑ…Ñ”½˜Ñ¡”±…ÍÐ½µÁÕÑ•Ñ½­•¸°Í¼Ý”É•ÑÕÉ¸4(€€€€€€€¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì€´€Ä¸4(€€€€€€€€ˆˆˆ4(€€€€€€€É•ÑÕÉ¸¹Õµ}½µÁÕÑ•‘}Ñ½­•¹Ì€´€Ä4(4(€€€‘•˜…¡•}‰±½­Ì¡Í•±˜°É•ÅÕ•ÍÐèI•ÅÕ•ÍÐ°¹Õµ}Ñ½­•¹Ìè¥¹Ð¤€´ø9½¹”è4(€€€€€€€¹Õµ}…¡•‘}‰±½­Í}‰•™½É”€ôÍ•±˜¹¹Õµ}…¡•‘}‰±½¬¹•Ð¡É•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥°€À¤4(€€€€€€€ÍÕÁ•È ¤¹…¡•}‰±½­Ì¡É•ÅÕ•ÍÐ°¹Õµ}Ñ½­•¹Ì¤4(€€€€€€€¹Õµ}…¡•‘}‰±½­Í}…™Ñ•È€ôÍ•±˜¹¹Õµ}…¡•‘}‰±½¬¹•Ð¡É•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥°€À¤4(€€€€€€€¥˜¹Õµ}…¡•‘}‰±½­Í}…™Ñ•È€ø¹Õµ}…¡•‘}‰±½­Í}‰•™½É”è4(€€€€€€€€€€€™½È‰±½¬¥¸Í•±˜¹É•Å}Ñ½}‰±½­ÍmÉ•ÅÕ•ÍÐ¹É•ÅÕ•ÍÑ}¥‘ul4(€€€€€€€€€€€€€€€¹Õµ}…¡•‘}‰±½­Í}‰•™½É”é¹Õµ}…¡•‘}‰±½­Í}…™Ñ•È4(€€€€€€€€€€€tè4(€€€€€€€€€€€€€€€¥˜‰±½¬¹¥Í}¹Õ±°è4(€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”4(€€€€€€€€€€€€€€€…ÍÍ•ÉÐ‰±½¬¹‰±½­}¡…Í ¥Ì¹½Ð9½¹”4(€€€€€€€€€€€€€€€Í•±˜¹…¡•‘}‰±½­Í}Ñ¡¥Í}ÍÑ•À¹…‘¡‰±½¬¹‰±½­}¡…Í ¤4(4(€€€‘•˜¹•Ý}ÍÑ•Á}ÍÑ…ÉÑÌ¡Í•±˜¤€´ø9½¹”è4(€€€€€€€Í•±˜¹…¡•‘}‰±½­Í}Ñ¡¥Í}ÍÑ•À¹±•…È ¤4(4(4)±…ÍÌÉ½ÍÍÑÑ•¹Ñ¥½¹5…¹…•È¡M¥¹±•QåÁ•-Y…¡•5…¹…•È¤è4(€€€€ˆˆ‰5…¹…•È™½ÈÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸-X…¡”¥¸•¹½‘•Èµ‘•½‘•Èµ½‘•±Ì¸ˆˆˆ4(4(€€€‘•˜…‘‘}±½…±}½µÁÕÑ•‘}‰±½­Ì 4(€€€€€€€Í•±˜°4(€€€€€€€É•ÅÕ•ÍÑ}¥èÍÑÈ°4(€€€€€€€¹•Ý}½µÁÕÑ•‘}‰±½­ÌèM•ÅÕ•¹•m-Y…¡•	±½­t°4(€€€€€€€¹Õµ}±½…±}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€€€€€¹Õµ}•áÑ•É¹…±}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€€¤€´ø9½¹”è4(€€€€€€€€Œ]”‘¼¹½Ð…¡”‰±½­Ì™½ÈÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸Ñ¼‰”Í¡…É•‰•ÑÝ••¸4(€€€€€€€€ŒÉ•ÅÕ•ÍÑÌ°Í¼€¹•Ý}½µÁÕÑ•‘}‰±½­Í€Í¡½Õ±…±Ý…åÌ‰”•µÁÑä¸4(€€€€€€€…ÍÍ•ÉÐ±•¸¡¹•Ý}½µÁÕÑ•‘}‰±½­Ì¤€ôô€À4(4(€€€‘•˜…±±½…Ñ•}•áÑ•É¹…±}½µÁÕÑ•‘}‰±½­Ì 4(€€€€€€€Í•±˜°4(€€€€€€€É•ÅÕ•ÍÑ}¥èÍÑÈ°4(€€€€€€€¹Õµ}±½…±}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€€€€€¹Õµ}•áÑ•É¹…±}½µÁÕÑ•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€€¤€´ø9½¹”è4(€€€€€€€€ŒÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸ÍÑ…Ñ•Ì…É”É•ÅÕ•ÍÐµÍÁ•¥™¥Œ…¹…É”¹•Ù•È±½…‘•…Ì4(€€€€€€€€ŒÉ•ÕÍ…‰±”•áÑ•É¹…°ÁÉ•™¥à-X¸4(€€€€€€€É•ÑÕÉ¸4(4(€€€‘•˜…¡•}‰±½­Ì¡Í•±˜°É•ÅÕ•ÍÐèI•ÅÕ•ÍÐ°¹Õµ}Ñ½­•¹Ìè¥¹Ð¤€´ø9½¹”è4(€€€€€€€€Œ]”‘¼¹½Ð…¡”‰±½­Ì™½ÈÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸Ñ¼‰”Í¡…É•‰•ÑÝ••¸4(€€€€€€€€ŒÉ•ÅÕ•ÍÑÌ°Í¼Ñ¡¥Ìµ•Ñ¡½¥Ì¹½ÐÉ•±•Ù…¹Ð¸4(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰M¡½Õ±¹½Ð‰”…±±•…ÌÁÉ•™¥à…¡¥¹œ¥Ì‘¥Í…‰±•¸ˆ¤4(4(€€€‘•˜•Ñ}¹Õµ}½µµ½¹}ÁÉ•™¥á}‰±½­Ì¡Í•±˜°ÉÕ¹¹¥¹}É•ÅÕ•ÍÑ}¥èÍÑÈ¤€´ø¥¹Ðè4(€€€€€€€€ŒÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸‰±½­Ì½¹Ñ…¥¸É•ÅÕ•ÍÐµÍÁ•¥™¥Œ•¹½‘•ÈÍÑ…Ñ•Ì4(€€€€€€€€Œ…¹…É”¹½ÐÍ¡…É•‰•ÑÝ••¸‘¥™™•É•¹ÐÉ•ÅÕ•ÍÑÌ4(€€€€€€€É•ÑÕÉ¸€À4(4(€€€±…ÍÍµ•Ñ¡½4(€€€‘•˜™¥¹‘}±½¹•ÍÑ}…¡•}¡¥Ð 4(€€€€€€€±Ì°4(€€€€€€€‰±½­}¡…Í¡•Ìè	±½­!…Í¡1¥ÍÐ°4(€€€€€€€µ…á}±•¹Ñ è¥¹Ð°4(€€€€€€€­Ù}…¡•}É½ÕÁ}¥‘Ìè±¥ÍÑm¥¹Ñt°4(€€€€€€€‰±½­}Á½½°è	±½­A½½°°4(€€€€€€€­Ù}…¡•}ÍÁ•Œè-Y…¡•MÁ•Œ°4(€€€€€€€ÕÍ•}•…±”è‰½½°°4(€€€€€€€…±¥¹µ•¹Ñ}Ñ½­•¹Ìè¥¹Ð°4(€€€€€€€‘Á}Ý½É±‘}Í¥é”è¥¹Ð€ô€Ä°4(€€€€€€€ÁÁ}Ý½É±‘}Í¥é”è¥¹Ð€ô€Ä°4(€€€€¤€´øÑÕÁ±•m±¥ÍÑm-Y…¡•	±½­t°€¸¸¹tè4(€€€€€€€…ÍÍ•ÉÐ¥Í¥¹ÍÑ…¹”¡­Ù}…¡•}ÍÁ•Œ°É½ÍÍÑÑ•¹Ñ¥½¹MÁ•Œ¤°€ 4(€€€€€€€€€€€€‰É½ÍÍÑÑ•¹Ñ¥½¹5…¹…•È…¸½¹±ä‰”ÕÍ•™½ÈÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸É½ÕÁÌˆ4(€€€€€€€€¤4(€€€€€€€€ŒÉ½ÍÌµ…ÑÑ•¹Ñ¥½¸‘½•Ì¹½Ð‰•¹•™¥Ð™É½´ÁÉ•™¥à…¡¥¹œÍ¥¹”è4(€€€€€€€€Œ€Ä¸¹½‘•ÈÍÑ…Ñ•Ì…É”Õ¹¥ÅÕ”Á•ÈÉ•ÅÕ•ÍÐ€¡‘¥™™•É•¹Ð…Õ‘¥¼½¥µ…”4(€€€€€€€€Œ€€€¥¹ÁÕÑÌ¤4(€€€€€€€€Œ€È¸¹½‘•ÈÍÑ…Ñ•Ì…É”½µÁÕÑ•½¹”Á•ÈÉ•ÅÕ•ÍÐ°¹½Ð¥¹É•µ•¹Ñ…±±ä4(€€€€€€€€Œ€Ì¸9¼É•ÕÍ…‰±”ÁÉ•™¥à•á¥ÍÑÌ‰•ÑÝ••¸‘¥™™•É•¹ÐµÕ±Ñ¥µ½‘…°¥¹ÁÕÑÌ4(€€€€€€€€ŒI•ÑÕÉ¸•µÁÑä‰±½­ÌÑ¼¥¹‘¥…Ñ”¹¼…¡”¡¥ÑÌ4(€€€€€€€É…¥Í”9½Ñ%µÁ±•µ•¹Ñ•‘ÉÉ½È ‰É½ÍÍÑÑ•¹Ñ¥½¹5…¹…•È‘½•Ì¹½ÐÍÕÁÁ½ÉÐ…¡¥¹œˆ¤4(4(4)±…ÍÌM¥¹­Õ±±ÑÑ•¹Ñ¥½¹5…¹…•È¡Õ±±ÑÑ•¹Ñ¥½¹5…¹…•È¤è4(€€€‘•˜}}¥¹¥Ñ}| 4(€€€€€€€Í•±˜°4(€€€€€€€­Ù}…¡•}ÍÁ•ŒèM¥¹­Õ±±ÑÑ•¹Ñ¥½¹MÁ•Œ°4(€€€€€€€‰±½­}Á½½°è	±½­A½½°°4(€€€€€€€•¹…‰±•}…¡¥¹œè‰½½°°4(€€€€€€€­Ù}…¡•}É½ÕÁ}¥è¥¹Ð°4(€€€€€€€‘Á}Ý½É±‘}Í¥é”è¥¹Ð€ô€Ä°4(€€€€€€€ÁÁ}Ý½É±‘}Í¥é”è¥¹Ð€ô€Ä°4(€€€€¤è4(€€€€€€€ÍÕÁ•È ¤¹}}¥¹¥Ñ}| 4(€€€€€€€€€€€­Ù}…¡•}ÍÁ•Œ°4(€€€€€€€€€€€‰±½­}Á½½°°4(€€€€€€€€€€€•¹…‰±•}…¡¥¹œ°4(€€€€€€€€€€€­Ù}…¡•}É½ÕÁ}¥°4(€€€€€€€€€€€‘Á}Ý½É±‘}Í¥é”°4(€€€€€€€€€€€ÁÁ}Ý½É±‘}Í¥é”°4(€€€€€€€€¤4(€€€€€€€Í¥¹­}±•¸€ô­Ù}…¡•}ÍÁ•Œ¹Í¥¹­}±•¸4(€€€€€€€…ÍÍ•ÉÐÍ¥¹­}±•¸¥Ì¹½Ð9½¹”…¹Í¥¹­}±•¸€ø€À…¹Í¥¹­}±•¸€”Í•±˜¹‰±½­}Í¥é”€ôô€À4(€€€€€€€¹Õµ}Í¥¹­}‰±½¬€ôÍ¥¹­}±•¸€¼¼Í•±˜¹‰±½­}Í¥é”4(€€€€€€€Í•±˜¹Í¥¹­}‰±½­Ì€ôÍ•±˜¹‰±½­}Á½½°¹™É••}‰±½­}ÅÕ•Õ”¹Á½Á±•™Ñ}¸¡¹Õµ}Í¥¹­}‰±½¬¤4(4(4)ÍÁ•}µ…¹…•É}µ…Àè‘¥ÑmÑåÁ•m-Y…¡•MÁ•t°ÑåÁ•mM¥¹±•QåÁ•-Y…¡•5…¹…•Éut€ôì4(€€€Õ±±ÑÑ•¹Ñ¥½¹MÁ•ŒèÕ±±ÑÑ•¹Ñ¥½¹5…¹…•È°4(€€€QEÕ±±ÑÑ•¹Ñ¥½¹MÁ•ŒèÕ±±ÑÑ•¹Ñ¥½¹5…¹…•È°4(€€€51ÑÑ•¹Ñ¥½¹MÁ•ŒèÕ±±ÑÑ•¹Ñ¥½¹5…¹…•È°4(€€€M±¥‘¥¹]¥¹‘½ÝMÁ•ŒèM±¥‘¥¹]¥¹‘½Ý5…¹…•È°4(€€€M±¥‘¥¹]¥¹‘½Ý51MÁ•ŒèM±¥‘¥¹]¥¹‘½Ý5…¹…•È°4(€€€¡Õ¹­•‘1½…±ÑÑ•¹Ñ¥½¹MÁ•Œè¡Õ¹­•‘1½…±ÑÑ•¹Ñ¥½¹5…¹…•È°4(€€€5…µ‰…MÁ•Œè5…µ‰…5…¹…•È°4(€€€É½ÍÍÑÑ•¹Ñ¥½¹MÁ•ŒèÉ½ÍÍÑÑ•¹Ñ¥½¹5…¹…•È°4(€€€M¥¹­Õ±±ÑÑ•¹Ñ¥½¹MÁ•ŒèM¥¹­Õ±±ÑÑ•¹Ñ¥½¹5…¹…•È°4)ô4(4(4)‘•˜•Ñ}µ…¹…•É}™½É}­Ù}…¡•}ÍÁ•Œ 4(€€€­Ù}…¡•}ÍÁ•Œè-Y…¡•MÁ•Œ°4(€€€µ…á}¹Õµ}‰…Ñ¡•‘}Ñ½­•¹Ìè¥¹Ð°4(€€€µ…á}µ½‘•±}±•¸è¥¹Ð°4(€€€€¨©­Ý…ÉÌ°4(¤€´øM¥¹±•QåÁ•-Y…¡•5…¹…•Èè4(€€€µ…¹…•É}±…ÍÌ€ôÍÁ•}µ…¹…•É}µ…ÁmÑåÁ”¡­Ù}…¡•}ÍÁ•Œ¥t4(€€€€ŒM±¥‘¥¹]¥¹‘½Ü€¼¡Õ¹­•‘1½…±ÑÑ•¹Ñ¥½¸µ…¹…•ÉÌÉ•å±”‰±½­Ì…É½ÍÌ4(€€€€Œ¡Õ¹­ÌìÑ¡”ÉÕ¹Ñ¥µ”…‘µ¥ÍÍ¥½¸…ÀµÕÍÐµ…Ñ Ñ¡”É•å±¥¹œµ…Ý…É”‰½Õ¹4(€€€€ŒÑ¡”ÍÑ…ÉÑÕÀÁ½½°Í¥é•ÈÕÍ•Ì€¡Í¥¹±”Í½ÕÉ”½˜ÑÉÕÑ èÑ¡”ÍÁ•Œµ•Ñ¡½¤¸4(€€€¥˜¥Í¥¹ÍÑ…¹”¡­Ù}…¡•}ÍÁ•Œ°€¡M±¥‘¥¹]¥¹‘½ÝMÁ•Œ°¡Õ¹­•‘1½…±ÑÑ•¹Ñ¥½¹MÁ•Œ¤¤è4(€€€€€€€­Ý…ÉÍl‰µ…á}…‘µ¥ÍÍ¥½¹}‰±½­Í}Á•É}É•ÅÕ•ÍÐ‰t€ô€ 4(€€€€€€€€€€€­Ù}…¡•}ÍÁ•Œ¹µ…á}…‘µ¥ÍÍ¥½¹}‰±½­Í}Á•É}É•ÅÕ•ÍÐ 4(€€€€€€€€€€€€€€€µ…á}¹Õµ}‰…Ñ¡•‘}Ñ½­•¹Ìõµ…á}¹Õµ}‰…Ñ¡•‘}Ñ½­•¹Ì°4(€€€€€€€€€€€€€€€µ…á}µ½‘•±}±•¸õµ…á}µ½‘•±}±•¸°4(€€€€€€€€€€€€¤4(€€€€€€€€¤4(€€€µ…¹…•È€ôµ…¹…•É}±…ÍÌ¡­Ù}…¡•}ÍÁ•Œ°€¨©­Ý…ÉÌ¤4(€€€É•ÑÕÉ¸µ…¹…•È4(