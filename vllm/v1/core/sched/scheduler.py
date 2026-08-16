# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.utils.math_utils import round_down
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None and hasattr(
            self.connector, "bind_gpu_block_pool"
        ):
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.retain_mamba_align_mtp_cache_block = (
            speculative_config is not None
            and speculative_config.method == "mtp"
            and self.need_mamba_block_aligned_split
            and envs.VLLM_MAMBA_ALIGN_RETAIN_MTP_CACHE_BLOCK
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self._pause_state: PauseState = PauseState.UNPAUSED

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # In align mode a cached Mamba block represents recurrent state after
        # exactly (block_index + 1) * block_size tokens. Therefore every
        # non-final prefill chunk must end on a block boundary. The final tail
        # may remain unaligned because it is not reused as a boundary state.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if num_computed_tokens < prefill_end:
            block_size = self.cache_config.block_size

            # EAGLE drops the last matched full-attention block. Keep the last
            # reusable Mamba boundary one block earlier so all hybrid groups
            # describe the same prefix length.
            last_cache_position = round_down(request.num_tokens, block_size)
            # MTP follows the EAGLE scheduler path, but an uncached prompt tail
            # still runs and produces the hidden states needed by the proposer.
            # In that case the final aligned Mamba state is valid and retaining
            # it avoids throwing away a full prefix block.
            retain_final_mtp_block = (
                getattr(self, "retain_mamba_align_mtp_cache_block", False)
                and last_cache_position < request.num_tokens
            )
            if self.use_eagle and not retain_final_mtp_block:
                last_cache_position = max(last_cache_position - block_size, 0)

            chunk_end = num_computed_tokens + num_new_tokens
            if num_computed_tokens < last_cache_position:
                chunk_end = min(
                    round_down(chunk_end, block_size), last_cache_position
                )
            elif chunk_end < prefill_end:
                chunk_end = round_down(chunk_end, block_size)

            num_new_tokens = max(chunk_end - num_computed_tokens, 0)

        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is genÛ|ÖÚ$z{-®éÜj×·eö66†UöWf–7F–öåöWfVçG3ÖWf–7F–öåöWfVçG2ÀĞ¢7V5öFV6öF–æu÷7FG3×7V5÷7FG2ÀĞ¢·eö6öææV7F÷%÷7FG3Ö6öææV7F÷%÷7FG5÷–ÆöBÀĞ¢7VFw&…÷7FG3Ö7VFw&…÷7FG2ÀĞ¢W&e÷7FG3×W&e÷7FG2ÀĞ¢Ğ Ğ¢FVbÖ¶U÷7V5öFV6öF–æu÷7FG2€Ğ¢6VÆbÀĞ¢7V5öFV6öF–æu÷7FG3¢7V4FV6öF–æu7FG2ÂæöæRÀĞ¢çVÕöG&gE÷Fö¶Vç3¢–çBÀĞ¢çVÕö66WFVE÷Fö¶Vç3¢–çBÀĞ¢çVÕö–çfÆ–E÷7V5÷Fö¶Vç3¢F–7E·7G"Â–çEÒÂæöæRÀĞ¢&WVW7Eö–C¢7G"ÀĞ¢’Óâ7V4FV6öF–æu7FG2ÂæöæS Ğ¢–bæ÷B6VÆbæÆöu÷7FG2÷"æ÷BçVÕöG&gE÷Fö¶Vç3 Ğ¢&WGW&âæöæPĞ¢–b7V5öFV6öF–æu÷7FG2—2æöæS Ğ¢7V5öFV6öF–æu÷7FG2Ò7V4FV6öF–æu7FG2ææWr‡6VÆbæçVÕ÷7V5÷Fö¶Vç2Ğ¢–bçVÕö–çfÆ–E÷7V5÷Fö¶Vç3 Ğ¢çVÕöG&gE÷Fö¶Vç2ÓÒçVÕö–çfÆ–E÷7V5÷Fö¶Vç2ævWB‡&WVW7Eö–BÂĞ¢7V5öFV6öF–æu÷7FG2æö'6W'fUöG&gB€Ğ¢çVÕöG&gE÷Fö¶Vç3ÖçVÕöG&gE÷Fö¶Vç2ÂçVÕö66WFVE÷Fö¶Vç3ÖçVÕö66WFVE÷Fö¶Vç0Ğ¢Ğ¢&WGW&â7V5öFV6öF–æu÷7FG0Ğ Ğ¢FVb6‡WFF÷vâ‡6VÆb’ÓâæöæS Ğ¢–b6VÆbæ·eöWfVçE÷V&Æ—6†W# Ğ¢6VÆbæ·eöWfVçE÷V&Æ—6†W"ç6‡WFF÷vâ‚Ğ¢–b6VÆbæ6öææV7F÷"—2æ÷BæöæS Ğ¢6VÆbæ6öææV7F÷"ç6‡WFF÷vâ‚Ğ Ğ¢222222222222222222222222222222222222222222222222222222222222222222222220Ğ¢2µb6öææV7F÷"&VÆFVBÖWF†öG0Ğ¢222222222222222222222222222222222222222222222222222222222222222222222220Ğ Ğ¢FVbvWEö·eö6öææV7F÷"‡6VÆb’Óâµd6öææV7F÷$&6UõcÂæöæS Ğ¢&WGW&â6VÆbæ6öææV7F÷ Ğ Ğ¢FVbö6öææV7F÷%öf–æ—6†VB€Ğ¢6VÆbÂ&WVW7C¢&WVW7@Ğ¢’ÓâGWÆU¶&ööÂÂF–7E·7G"Âç•ÒÂæöæUÓ Ğ¢"" Ğ¢–çfö¶RF†Rµb6öææV7F÷"&WVW7Eöf–æ—6†VB‚’ÖWF†öB–bÆ–6&ÆRàĞ Ğ¢&WGW&ç2÷F–öæÂ·bG&ç6fW"&ÖWFW'2Fò&R–æ6ÇVFVBv—F‚F†PĞ¢&WVW7B÷WGWG2àĞ¢"" Ğ¢–b6VÆbæ6öææV7F÷"—2æöæS Ğ¢&WGW&âfÇ6RÂæöæPĞ Ğ¢2g&VRç’÷WBÖöb×v–æF÷r&Vf—‚&Æö6·2&Vf÷&RvR†æBF†R&Æö6²F&ÆRFğĞ¢2F†R6öææV7F÷"àĞ¢6VÆbæ·eö66†UöÖævW"ç&VÖ÷fU÷6¶—VEö&Æö6·2€Ğ¢&WVW7Eö–C×&WVW7Bç&WVW7Eö–BÀĞ¢F÷FÅö6ö×WFVE÷Fö¶Vç3×&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2ÀĞ¢Ğ Ğ¢&Æö6µö–G2Ò6VÆbæ·eö66†UöÖævW"ævWEö&Æö6µö–G2‡&WVW7Bç&WVW7Eö–BĞ Ğ¢–bæ÷B—6–ç7Fæ6R‡6VÆbæ6öææV7F÷"Â7W÷'G4„Ô“ Ğ¢2äõDR„·VçF’“¢vR6†÷VÆBFW&V6FRF†—26öFRF‚gFW"vRVæf÷&6PĞ¢2ÆÂ6öææV7F÷'2Fò7W÷'B„ÔàĞ¢2‡–'&–BÖVÖ÷'’ÆÆö6F÷"6†÷VÆB&RÇ&VG’GW&æVBöfbf÷"F†—0Ğ¢26öFRF‚Â'WBÆWBw2F÷V&ÆRÖ6†V6²†W&RàĞ¢76W'BÆVâ‡6VÆbæ·eö66†Uö6öæf–ræ·eö66†Uöw&÷W2’ÓÒĞ¢&WGW&â6VÆbæ6öææV7F÷"ç&WVW7Eöf–æ—6†VB‡&WVW7BÂ&Æö6µö–G5³ÒĞ Ğ¢&WGW&â6VÆbæ6öææV7F÷"ç&WVW7Eöf–æ—6†VEöÆÅöw&÷W2‡&WVW7BÂ&Æö6µö–G2Ğ Ğ¢FVb÷WFFU÷v—F–æuöf÷%÷&VÖ÷FUö·b‡6VÆbÂ&WVW7C¢&WVW7B’ÓâæöæS Ğ¢"" Ğ¢µb6öææV7F÷#¢WFFR&WVW7B7FFRgFW"7–æ2&V7b—2f–æ—6†VBàĞ Ğ¢v†VâF†R·bG&ç6fW"—2&VG’ÂvR66†RF†R&Æö6·0Ğ¢æBF†R&WVW7B7FFRv–ÆÂ&RÖ÷fVB&6²Fòt•D”ärg&öĞĞ¢t•D”äuôdõ%õ$TÔõDUôµbàĞ¢"" Ğ¢76W'B6VÆbæ6öææV7F÷"—2æ÷BæöæPĞ Ğ¢–b&WVW7Bç&WVW7Eö–B–â6VÆbæf–ÆVE÷&V7f–æuö·e÷&Wö–G3 Ğ¢2&WVW7B†BµbÆöBf–ÇW&W3²çVÕö6ö×WFVE÷Fö¶Vç2v2Ç&VGĞ¢2WFFVB–â÷WFFU÷&WVW7G5÷v—F…ö–çfÆ–Eö&Æö6·0Ğ¢–b&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç3 Ğ¢266†Rç’fÆ–B6ö×WFVBFö¶Vç2àĞ¢6VÆbæ·eö66†UöÖævW"æ66†Uö&Æö6·2‡&WVW7BÂ&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ğ¢VÇ6S Ğ¢2æòfÆ–B6ö×WFVBFö¶Vç2Â&VÆV6RÆÆö6FVB&Æö6·2àĞ¢2F†W&RÖ’&RÆö6Â66†R†—Böâ&WG'’àĞ¢6VÆbæ·eö66†UöÖævW"æg&VR‡&WVW7BĞ Ğ¢6VÆbæf–ÆVE÷&V7f–æuö·e÷&Wö–G2ç&VÖ÷fR‡&WVW7Bç&WVW7Eö–BĞ¢VÇ6S Ğ¢2æ÷rF†BF†R&Æö6·2&R&VG’Â7GVÆÇ’66†RF†VÒàĞ¢2F†—2v–ÆÂ66†RF†R&Æö6·2–fb66†–ær—2Væ&ÆVBàĞ¢6VÆbæ·eö66†UöÖævW"æ66†Uö&Æö6·2‡&WVW7BÂ&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ğ Ğ¢2öâgVÆÂ&ö×B†—BÂvRæVVBFò&RÖ6ö×WFRF†RÆ7BFö¶VàĞ¢2–â÷&FW"Fò&R&ÆRFò6×ÆRF†RæW‡BFö¶VàĞ¢–b&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2ÓÒ&WVW7BæçVÕ÷Fö¶Vç3 Ğ¢&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ò&WVW7BæçVÕ÷Fö¶Vç2ÒĞ Ğ¢6VÆbæf–æ—6†VE÷&V7f–æuö·e÷&Wö–G2ç&VÖ÷fR‡&WVW7Bç&WVW7Eö–BĞ Ğ¢FVb÷G'•÷&öÖ÷FUö&Æö6¶VE÷v—F–æu÷&WVW7B‡6VÆbÂ&WVW7C¢&WVW7B’Óâ&ööÃ Ğ¢"" Ğ¢G'’Fò&öÖ÷FR&Æö6¶VBv—F–ær&WVW7B&6²Fò66†VGVÆ&ÆR7FFW2àĞ¢"" Ğ¢–b&WVW7Bç7FGW2ÓÒ&WVW7E7FGW2åt•D”äuôdõ%õ$TÔõDUôµe3 Ğ¢2f–æ—6†VE÷&V7f–æuö·e÷&Wö–G2—2÷VÆFVBGW&–æpĞ¢2WFFUög&öÕö÷WGWB‚’Â&6VBöâv÷&¶W"×6–FR6öææV7F÷"6–væÇ0Ğ¢2–âµd6öææV7F÷$÷WGWBæf–æ—6†VE÷&V7f–æpĞ¢–b&WVW7Bç&WVW7Eö–Bæ÷B–â6VÆbæf–æ—6†VE÷&V7f–æuö·e÷&Wö–G3 Ğ¢&WGW&âfÇ6PĞ¢6VÆbå÷WFFU÷v—F–æuöf÷%÷&VÖ÷FUö·b‡&WVW7BĞ¢–b&WVW7BæçVÕ÷&VV×F–öç3 Ğ¢&WVW7Bç7FGW2Ò&WVW7E7FGW2å$TTÕDT@Ğ¢VÇ6S Ğ¢&WVW7Bç7FGW2Ò&WVW7E7FGW2åt•D”äpĞ¢&WGW&âG'VPĞ Ğ¢–b&WVW7Bç7FGW2ÓÒ&WVW7E7FGW2åt•D”äuôdõ%õ5E%T5EU$TEôõUEUEôu$ÔÔ# Ğ¢7G'V7GW&VEö÷WGWE÷&WÒ&WVW7Bç7G'V7GW&VEö÷WGWE÷&WVW7@Ğ¢–bæ÷B‡7G'V7GW&VEö÷WGWE÷&WæB7G'V7GW&VEö÷WGWE÷&Wæw&ÖÖ"“ Ğ¢&WGW&âfÇ6PĞ¢&WVW7Bç7FGW2Ò&WVW7E7FGW2åt•D”äpĞ¢&WGW&âG'VPĞ Ğ¢–b&WVW7Bç7FGW2ÓÒ&WVW7E7FGW2åt•D”äuôdõ%õ5E$TÔ”äuõ$U Ğ¢76W'Bæ÷B&WVW7Bç7G&VÖ–æu÷VWVPĞ¢&WGW&âfÇ6PĞ Ğ¢&—6R76W'F–öäW'&÷"€Ğ¢%VæW‡V7FVB&Æö6¶VBv—F–ær7FGW2–â&öÖ÷F–öã¢ Ğ¢b'·&WVW7Bç7FGW2ææÖWÒf÷"&WVW7B·&WVW7Bç&WVW7Eö–GÒ Ğ¢Ğ Ğ¢FVb÷WFFUög&öÕö·e÷†fW%öf–æ—6†VB‡6VÆbÂ·eö6öææV7F÷%ö÷WGWC¢µd6öææV7F÷$÷WGWB“ Ğ¢"" Ğ¢µb6öææV7F÷#¢WFFRF†R66†VGVÆW"7FFR&6VBöâF†R÷WGWBàĞ Ğ¢F†Rv÷&¶W"6–FR6öææV7F÷'2FBf–æ—6†VE÷&V7f–æræ@Ğ¢f–æ—6†VE÷6VæF–ær&W2FòF†R÷WGWBàĞ¢¢–bf–æ—6†VE÷6VæF–æs¢g&VRF†R&Æö6·0Ğ¢2–bf–æ—6†VE÷&V7f–æs¢FBFò7FFR6òvR6àĞ¢66†VGVÆRF†R&WVW7BGW&–ærF†RæW‡B7FWàĞ¢"" Ğ Ğ¢–b6VÆbæ6öææV7F÷"—2æ÷BæöæS Ğ¢6VÆbæ6öææV7F÷"çWFFUö6öææV7F÷%ö÷WGWB†·eö6öææV7F÷%ö÷WGWBĞ Ğ¢2µb6öææV7F÷#£¢WFFR&V7bæB6VæB7FGW2g&öÒÆ7B7FWàĞ¢f÷"&Wö–B–â·eö6öææV7F÷%ö÷WGWBæf–æ—6†VE÷&V7f–ær÷"‚“ Ğ¢ÆövvW"æFV'Vr‚$f–æ—6†VB&V7f–ærµbG&ç6fW"f÷"&WVW7BW2"Â&Wö–BĞ¢76W'B&Wö–B–â6VÆbç&WVW7G0Ğ¢&WÒ6VÆbç&WVW7G5·&Wö–EĞĞ¢–b&Wç7FGW2ÓÒ&WVW7E7FGW2åt•D”äuôdõ%õ$TÔõDUôµe3 Ğ¢6VÆbæf–æ—6†VE÷&V7f–æuö·e÷&Wö–G2æFB‡&Wö–BĞ¢VÇ6S Ğ¢76W'B&WVW7E7FGW2æ—5öf–æ—6†VB‡&Wç7FGW2Ğ¢6VÆbåög&VUö&Æö6·2‡6VÆbç&WVW7G5·&Wö–EÒĞ¢f÷"&Wö–B–â·eö6öææV7F÷%ö÷WGWBæf–æ—6†VE÷6VæF–ær÷"‚“ Ğ¢ÆövvW"æFV'Vr‚$f–æ—6†VB6VæF–ærµbG&ç6fW"f÷"&WVW7BW2"Â&Wö–BĞ¢76W'B&Wö–B–â6VÆbç&WVW7G0Ğ¢6VÆbåög&VUö&Æö6·2‡6VÆbç&WVW7G5·&Wö–EÒĞ Ğ¢FVb÷WFFU÷&WVW7G5÷v—F…ö–çfÆ–Eö&Æö6·2€Ğ¢6VÆbÀĞ¢&WVW7G3¢—FW&&ÆUµ&WVW7EÒÀĞ¢–çfÆ–Eö&Æö6µö–G3¢6WE¶–çEÒÀĞ¢çVÕ÷66†VGVÆVE÷Fö¶Vç3¢F–7E·7G"Â–çEÒÀĞ¢Wf–7Eö&Æö6·3¢&ööÂÒG'VRÀĞ¢’ÓâGWÆU·6WE·7G%ÒÂ–çBÂ6WE¶–çEÕÓ Ğ¢"" Ğ¢–FVçF–g’æBWFFR&WVW7G2ffV7FVB'’–çfÆ–Bµb66†R&Æö6·2àĞ Ğ¢F†—2ÖWF†öB66ç2F†Rv—fVâ&WVW7G2ÂFWFV7G2F†÷6Rv—F‚–çfÆ–B&Æö6·0Ğ¢æBF§W7G2F†V—"çVÕö6ö×WFVE÷Fö¶Vç6FòF†RÆöævW7BfÆ–B&Vf—‚àĞ¢f÷"ö'6W'f&–Æ—G’Â—BÇ6ò67V×VÆFW2F†RF÷FÂçVÖ&W"öbFö¶Vç2F†@Ğ¢v–ÆÂæVVBFò&R&V6ö×WFVB7&÷72ÆÂffV7FVB&WVW7G2àĞ Ğ¢&w3 Ğ¢&WVW7G3¢F†R6WBöb&WVW7G2Fò66âf÷"–çfÆ–B&Æö6·2àĞ¢–çfÆ–Eö&Æö6µö–G3¢”G2öb–çfÆ–B&Æö6·2àĞ¢çVÕ÷66†VGVÆVE÷Fö¶Vç3¢&Wö–BÓâçVÖ&W"öb66†VGVÆVBFö¶Vç2àĞ¢Wf–7Eö&Æö6·3¢v†WF†W"Fò6öÆÆV7B&Æö6·2f÷"Wf–7F–öâ„fÇ6Rf÷ Ğ¢7–æ2&WVW7G2v†–6‚&VâwB66†VB–WB’àĞ Ğ¢&WGW&ç3 Ğ¢GWÆS Ğ¢ÒffV7FVE÷&Wö–G2‡6WE·7G%Ò“¢”G2öb&WVW7G2–×7FVB'Ğ¢–çfÆ–B&Æö6·2àĞ¢ÒF÷FÅöffV7FVE÷Fö¶Vç2†–çB“¢F÷FÂçVÖ&W"öbFö¶Vç2F†B×W7@Ğ¢&R&V6ö×WFVB7&÷72ÆÂffV7FVB&WVW7G2àĞ¢Ò&Æö6·5÷FõöWf–7B‡6WE¶–çEÒ“¢&Æö6²”G2FòWf–7Bg&öÒ66†RÀĞ¢–æ6ÇVF–ær–çfÆ–B&Æö6·2æBF÷vç7G&VÒFWVæFVçB&Æö6·2àĞ¢"" Ğ¢ffV7FVE÷&Wö–G3¢6WE·7G%ÒÒ6WB‚Ğ¢F÷FÅöffV7FVE÷Fö¶Vç2Ò Ğ¢&Æö6·5÷FõöWf–7C¢6WE¶–çEÒÒ6WB‚Ğ¢2–b&Æö6²—2–çfÆ–BæB6†&VB'’×VÇF—ÆR&WVW7G2–âF†R&F6‚ÀĞ¢2F†W6R&WVW7G2×W7B&R&W66†VGVÆVBÂ'WBöæÇ’F†Rf—'7Bv–ÆÂ&V6ö×WFPĞ¢2—BâF†—26WBG&6·2&Æö6·2Ç&VG’Ö&¶VBf÷"&V6ö×WFF–öâàĞ¢Ö&¶VEö–çfÆ–Eö&Æö6µö–G3¢6WE¶–çEÒÒ6WB‚Ğ¢f÷"&WVW7B–â&WVW7G3 Ğ¢—5öffV7FVBÒfÇ6PĞ¢Ö&¶VEö–çfÆ–Eö&Æö6²ÒfÇ6PĞ¢&Wö–BÒ&WVW7Bç&WVW7Eö–@Ğ¢2DôDò†Ff–F"“¢FB7W÷'Bf÷"‡–'&–BÖVÖ÷'’ÆÆö6F÷ Ğ¢‡&Wö&Æö6µö–G2Â’Ò6VÆbæ·eö66†UöÖævW"ævWEö&Æö6µö–G2‡&Wö–BĞ¢2vR—FW&FRöæÇ’÷fW"&Æö6·2F†BÖ’6öçF–âW‡FW&æÆÇ’6ö×WFV@Ğ¢2Fö¶Vç0Ğ¢&WöçVÕö6ö×WFVE÷Fö¶Vç2Ò€Ğ¢&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2ÒçVÕ÷66†VGVÆVE÷Fö¶Vç2ævWB‡&Wö–BÂĞ¢Ğ Ğ¢&WöçVÕö6ö×WFVEö&Æö6·2Ò€Ğ¢&WöçVÕö6ö×WFVE÷Fö¶Vç2²6VÆbæ&Æö6µ÷6—¦RÒĞ¢’òò6VÆbæ&Æö6µ÷6—¦PĞ¢f÷"–G‚Â&Æö6µö–B–â¦—‡&ævR‡&WöçVÕö6ö×WFVEö&Æö6·2’Â&Wö&Æö6µö–G2“ Ğ¢–b&Æö6µö–Bæ÷B–â–çfÆ–Eö&Æö6µö–G3 Ğ¢6öçF–çVPĞ Ğ¢—5öffV7FVBÒG'VPĞ Ğ¢–b&Æö6µö–B–âÖ&¶VEö–çfÆ–Eö&Æö6µö–G3 Ğ¢2F†—2–çfÆ–B&Æö6²—26†&VBv—F‚&Wf–÷W2&WVW7@Ğ¢2æBv2Ç&VG’Ö&¶VBf÷"&V6ö×WFF–öâàĞ¢2F†—2ÖVç2F†—2&WVW7B6â7F–ÆÂ6öç6–FW"F†—2&Æö6°Ğ¢226ö×WFVBv†Vâ&W66†VGVÆVBàĞ¢27W'&VçFÇ’F†—2öæÇ’Æ–W2Fò7–æ2ÆöF–æs²7–æ0Ğ¢2ÆöF–ærFöW2æ÷B–WB7W÷'B&Æö6²6†&–æpĞ¢6öçF–çVPĞ Ğ¢Ö&¶VEö–çfÆ–Eö&Æö6µö–G2æFB†&Æö6µö–BĞ Ğ¢–bÖ&¶VEö–çfÆ–Eö&Æö6³ Ğ¢2F†—2&WVW7B†2Ç&VG’Ö&¶VBâ–çfÆ–B&Æö6²f÷ Ğ¢2&V6ö×WFF–öâæBWFFVB—G2çVÕö6ö×WFVE÷Fö¶Vç2àĞ¢6öçF–çVPĞ Ğ¢Ö&¶VEö–çfÆ–Eö&Æö6²ÒG'VPĞ¢2G'Væ6FRF†R6ö×WFVBFö¶Vç2BF†Rf—'7Bf–ÆVB&Æö6°Ğ¢&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ò–G‚¢6VÆbæ&Æö6µ÷6—¦PĞ¢çVÕöffV7FVE÷Fö¶Vç2Ò€Ğ¢&WöçVÕö6ö×WFVE÷Fö¶Vç2Ò&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç0Ğ¢Ğ¢F÷FÅöffV7FVE÷Fö¶Vç2³ÒçVÕöffV7FVE÷Fö¶Vç0Ğ Ğ¢26öÆÆV7B–çfÆ–B&Æö6²æBÆÂF÷vç7G&VÒFWVæFVçB&Æö6·0Ğ¢–bWf–7Eö&Æö6·3 Ğ¢&Æö6·5÷FõöWf–7BçWFFR‡&Wö&Æö6µö–G5¶–Gƒ¥ÒĞ Ğ¢–b—5öffV7FVC Ğ¢–bæ÷BÖ&¶VEö–çfÆ–Eö&Æö6³ Ğ¢2ÆÂ–çfÆ–B&Æö6·2öbF†—2&WVW7B&R6†&VBv—F€Ğ¢2&Wf–÷W2&WVW7G2æBv–ÆÂ&R&V6ö×WFVB'’F†VÒàĞ¢2&WfW'BFò6öç6–FW&–æröæÇ’66†VBFö¶Vç226ö×WFVBàĞ¢27W'&VçFÇ’F†—2öæÇ’Æ–W2Fò7–æ2ÆöF–æs²7–æ0Ğ¢2ÆöF–ærFöW2æ÷B–WB7W÷'B&Æö6²6†&–æpĞ¢F÷FÅöffV7FVE÷Fö¶Vç2³Ò€Ğ¢&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ò&WöçVÕö6ö×WFVE÷Fö¶Vç0Ğ¢Ğ¢&WVW7BæçVÕö6ö×WFVE÷Fö¶Vç2Ò&WöçVÕö6ö×WFVE÷Fö¶Vç0Ğ Ğ¢ffV7FVE÷&Wö–G2æFB‡&WVW7Bç&WVW7Eö–BĞ Ğ¢&WGW&âffV7FVE÷&Wö–G2ÂF÷FÅöffV7FVE÷Fö¶Vç2Â&Æö6·5÷FõöWf–7@Ğ Ğ¢FVbö†æFÆUö–çfÆ–Eö&Æö6·2€Ğ¢6VÆbÂ–çfÆ–Eö&Æö6µö–G3¢6WE¶–çEÒÂçVÕ÷66†VGVÆVE÷Fö¶Vç3¢F–7E·7G"Â–çEĞĞ¢’Óâ6WE·7G%Ó Ğ¢"" Ğ¢†æFÆR&WVW7G2ffV7FVB'’–çfÆ–Bµb66†R&Æö6·2àĞ Ğ¢&WGW&ç3 Ğ¢6WBöbffV7FVB&WVW7B”G2Fò6¶—–âWFFUög&öÕö÷WGWBÖ–âÆö÷àĞ¢"" Ğ¢6†÷VÆEöf–ÂÒæ÷B6VÆbç&V6ö×WFUö·eöÆöEöf–ÇW&W0Ğ Ğ¢2†æFÆR7–æ2µbÆöG2†æ÷B66†VB–WBÂWf–7Eö&Æö6·3ÔfÇ6RĞ¢7–æ5öÆöE÷&W2Ò€Ğ¢&WĞ¢f÷"&W–â6VÆbç6¶—VE÷v—F–æpĞ¢–b&Wç7FGW2ÓÒ&WVW7E7FGW2åt•D”äuôdõ%õ$TÔõDUôµe0Ğ¢Ğ¢7–æ5öf–ÆVE÷&Wö–G2ÂçVÕöf–ÆVE÷Fö¶Vç2ÂòÒ€Ğ¢6VÆbå÷WFFU÷&WVW7G5÷v—F…ö–çfÆ–Eö&Æö6·2€Ğ¢7–æ5öÆöE÷&W2ÀĞ¢–çfÆ–Eö&Æö6µö–G2ÀĞ¢çVÕ÷66†VGVÆVE÷Fö¶Vç2ÀĞ¢Wf–7Eö&Æö6·3ÔfÇ6RÀĞ¢Ğ¢Ğ Ğ¢F÷FÅöf–ÆVE÷&WVW7G2ÒÆVâ†7–æ5öf–ÆVE÷&Wö–G2Ğ¢F÷FÅöf–ÆVE÷Fö¶Vç2ÒçVÕöf–ÆVE÷Fö¶Vç0Ğ Ğ¢2†æFÆR7–æ2ÆöG2†Ö’&R66†VBÂ6öÆÆV7B&Æö6·2f÷"Wf–7F–öâĞ¢7–æ5öf–ÆVE÷&Wö–G2ÂçVÕöf–ÆVE÷Fö¶Vç2Â7–æ5ö&Æö6·5÷FõöWf–7BÒ€Ğ¢6VÆbå÷WFFU÷&WVW7G5÷v—F…ö–çfÆ–Eö&Æö6·2€Ğ¢6VÆbç'Vææ–ærÂ–çfÆ–Eö&Æö6µö–G2ÂçVÕ÷66†VGVÆVE÷Fö¶Vç2ÂWf–7Eö&Æö6·3ÕG'VPĞ¢Ğ¢Ğ Ğ¢F÷FÅöf–ÆVE÷&WVW7G2³ÒÆVâ‡7–æ5öf–ÆVE÷&Wö–G2Ğ¢F÷FÅöf–ÆVE÷Fö¶Vç2³ÒçVÕöf–ÆVE÷Fö¶Vç0Ğ Ğ¢–bæ÷BF÷FÅöf–ÆVE÷&WVW7G3 Ğ¢&WGW&â6WB‚Ğ Ğ¢2Wf–7B–çfÆ–B&Æö6·2æBF÷vç7G&VÒFWVæFVçB&Æö6·2g&öÒ66†PĞ¢2öæÇ’v†Vâæ÷BW6–ær&V6ö×WFRöÆ–7’‡v†W&R&Æö6·2v–ÆÂ&R&V6ö×WFV@Ğ¢2æB&WW6VB'’÷F†W"&WVW7G26†&–ærF†VÒĞ¢–b7–æ5ö&Æö6·5÷FõöWf–7BæBæ÷B6VÆbç&V6ö×WFUö·eöÆöEöf–ÇW&W3 Ğ¢6VÆbæ·eö66†UöÖævW"æWf–7Eö&Æö6·2‡7–æ5ö&Æö6·5÷FõöWf–7BĞ Ğ¢–b6†÷VÆEöf–Ã Ğ¢ÆÅöf–ÆVE÷&Wö–G2Ò7–æ5öf–ÆVE÷&Wö–G2Â7–æ5öf–ÆVE÷&Wö–G0Ğ¢ÆövvW"æW'&÷"€Ğ¢$f–Æ–ærVB&WVW7B‡2’GVRFòµbÆöBf–ÇW&R Ğ¢"†f–ÇW&U÷öÆ–7“Öf–ÂÂVBFö¶Vç2ffV7FVB’â&WVW7B”G3¢W2"ÀĞ¢F÷FÅöf–ÆVE÷&WVW7G2ÀĞ¢F÷FÅöf–ÆVE÷Fö¶Vç2ÀĞ¢ÆÅöf–ÆVE÷&Wö–G2ÀĞ¢Ğ¢&WGW&âÆÅöf–ÆVE÷&Wö–G0Ğ Ğ¢ÆövvW"çv&æ–ær€Ğ¢%&V6÷fW&VBg&öÒµbÆöBf–ÇW&S¢ Ğ¢"VB&WVW7B‡2’&W66†VGVÆVB‚VBFö¶Vç2ffV7FVB’â"ÀĞ¢F÷FÅöf–ÆVE÷&WVW7G2ÀĞ¢F÷FÅöf–ÆVE÷Fö¶Vç2ÀĞ¢Ğ Ğ¢2Ö&²7–æ2&WVW7G2v—F‚µbÆöBf–ÇW&W2f÷"&WG'’öæ6RÆöF–ær6ö×ÆWFW0Ğ¢6VÆbæf–ÆVE÷&V7f–æuö·e÷&Wö–G2ÃÒ7–æ5öf–ÆVE÷&Wö–G0Ğ¢2&WGW&â7–æ2ffV7FVB”G2Fò6¶—–âWFFUög&öÕö÷WGW@Ğ¢&WGW&â7–æ5öf–ÆVE÷&Wö–G0Ğ