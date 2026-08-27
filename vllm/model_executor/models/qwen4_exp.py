# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
"""Native vLLM execution support for Qwen4-Exp (Qwen3.8 Flash-Next).

Qwen4-Exp is not a Qwen3.5 alias: every decoder block operates on four
hyper-connection streams, the full-attention blocks carry a QSA indexer, and a
PLE block can inject a very large hashed n-gram embedding.  This module wires
the parts that have vLLM equivalents (GDN, MoE, TP/PP and CUDA Graph execution)
and keeps unsupported capacity-sensitive pieces explicit.
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    extract_layer_index,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)


class Qwen4ExpRMSNorm(GemmaRMSNorm):
    """Qwen4-Exp RMSNorm with optional per-stream grouping.

    Qwen4-Exp initializes norm weights to zero and applies ``weight + 1``;
    this is the convention already used by the Qwen3.5 SM75 implementation.
    """

    def __init__(self, hidden_size: int, eps: float, group_size: int | None = None):
        super().__init__(hidden_size, eps=eps)
        self.group_size = group_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.group_size is None:
            return super().forward(hidden_states)
        shape = hidden_states.shape
        if shape[-1] % self.group_size:
            raise ValueError(
                f"Qwen4-Exp grouped RMSNorm expects the final dimension to be "
                f"divisible by {self.group_size}, got {shape[-1]}"
            )
        x = hidden_states.reshape(*shape[:-1], -1, self.group_size)
        x = x * torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return x.reshape(shape) * (self.weight + 1)


class Qwen4ExpGatedResidual(nn.Module):
    """The low-rank multi-stream mixer used before/after each sub-block."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = config.hc_count * config.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(
            hc_hidden_size, eps=config.rms_norm_eps, group_size=config.hidden_size
        )
        # The down projection is row parallel because HF's TP plan shards its
        # input dimension.  The up projection gathers all low-rank outputs.
        self.input_mix_weight_down = RowParallelLinear(
            hc_hidden_size,
            config.hc_lowrank,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_mix_weight_down",
        )
        self.input_mix_weight_up = ColumnParallelLinear(
            config.hc_lowrank,
            hc_hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.input_mix_weight_up",
        )
        self.block_inject_weight = (
            ReplicatedLinear(
                hc_hidden_size,
                config.hc_count,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.block_inject_weight",
            )
            if use_combine
            else None
        )

    def forward(
        self, hyper_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection "
                f"features, got {hyper_input.shape[-1]}"
            )
        normed = self.hc_norm(hyper_input)
        down, _ = self.input_mix_weight_down(normed)
        down = torch.nn.functional.silu(down / self.hc_count)
        up, _ = self.input_mix_weight_up(down)
        mix = torch.sigmoid(up).view(-1, self.hc_count, self.hidden_size)
        mixed = (mix * normed.view(-1, self.hc_count, self.hidden_size)).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inject, _ = self.block_inject_weight(normed)
        inject = 2 * torch.sigmoid(inject / self.hc_count)
        return mixed, hyper_input, inject


class Qwen4ExpQSAIndexer(nn.Module):
    """Load and expose QSA indexer parameters.

    The generic vLLM paged-attention API has no per-query sparse-token mask
    input.  Until a QSA backend is selected, the native attention path uses a
    dense causal attention window and emits one explicit warning.  Keeping the
    indexer as a real TP-sharded module prevents silent missing-weight loads and
    makes a future QSA backend a local change.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        fields = (
            config.indexer_n_heads,
            config.indexer_kv_heads,
            config.indexer_head_dim,
            config.indexer_budget,
            config.indexer_compress_ratio,
        )
        if any(x is None for x in fields):
            raise ValueError(
                "Qwen4-Exp full-attention layers require complete QSA indexer "
                "configuration (indexer_n_heads/indexer_kv_heads/indexer_head_dim/"
                "indexer_budget/indexer_compress_ratio)"
            )
        self.indexer_n_heads = config.indexer_n_heads
        self.indexer_kv_heads = config.indexer_kv_heads
        self.indexer_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.index_qk_proj = ColumnParallelLinear(
            config.hidden_size,
            (config.indexer_n_heads + config.indexer_kv_heads)
            * config.indexer_head_dim,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj",
        )
        self.q_norm = Qwen4ExpRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = Qwen4ExpRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )


class Qwen4ExpSparseAttention(nn.Module):
    """Qwen4-Exp full-attention projections using vLLM paged attention.

    QSA metadata is retained and loaded, but the dense fallback is deliberate:
    passing an ordinary ``Attention`` object a fake sparse mask would produce
    incorrect results.  A future QSA implementation can replace this class
    without changing decoder or weight-loading code.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        model_config,
        cache_config,
        quant_config: QuantizationConfig | None,
        prefix: str,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.config = config
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads * 2,
            config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(config, quant_config, f"{prefix}.indexer")
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            layer_idx=extract_layer_index(prefix),
        )
        logger.warning_once(
            "Qwen4-Exp QSA indexer weights are loaded, but this build uses the "
            "dense paged-attention fallback because no QSA attention backend is "
            "available in vLLM 0.2.x."
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = self.q_norm(q).reshape(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).reshape(
            -1, self.kv_size
        )
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        out = out * torch.sigmoid(gate.reshape(-1, self.q_size))
        out, _ = self.o_proj(out)
        return out


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self, vllm_config: VllmConfig, layer_type: str, prefix: str = ""
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.layer_idx = extract_layer_index(prefix)
        self.layer_type = (
            "full_attention" if layer_type == "qwen_sparse_attention" else layer_type
        )
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        reduce_results = not (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=reduce_results,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen4ExpSparseAttention(
                config,
                vllm_config.model_config,
                vllm_config.cache_config,
                quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=reduce_results,
            )
        else:
            raise ValueError(f"Invalid Qwen4-Exp layer_type {layer_type!r}")
        self.mlp = Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config, prefix=f"{prefix}.mlp"
        )
        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            config, quant_config, f"{prefix}.attn_hyper_connection"
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            config, quant_config, f"{prefix}.mlp_hyper_connection"
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        mixed, hyper_input, injection_weights = self.attn_hyper_connection(
            hidden_states
        )
        if self.layer_type == "linear_attention":
            block_output = self.linear_attn(mixed)
        else:
            block_output = self.self_attn(positions, mixed)
        hidden_states = hyper_input + (
            block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        ).flatten(-2)

        mixed, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        block_output = self.mlp(mixed)
        return hyper_input + (
            block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        ).flatten(-2)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen4ExpModel(nn.Module):
    # Keep the same constituent-to-packed mappings as Qwen3.5.  In particular,
    # ModelOpt checkpoints expose routed/shared expert projections separately,
    # while the vLLM fused-MoE modules consume gate_up_proj.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_substr={
            ".indexer.q_layernorm": ".indexer.q_norm",
            ".indexer.k_layernorm": ".indexer.k_norm",
        },
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

        # The published checkpoint's PLE table is far larger than the target
        # adapters. Reject before allocating the decoder stack so a failed
        # launch cannot consume most of the available GPU memory first.
        if config.ple_layer_ids:
            raise RuntimeError(
                "Qwen4-Exp PLE is enabled by this checkpoint, but the native "
                "SM75 backend does not yet provide SSD/offloaded PLE embedding. "
                "The model cannot be loaded without a compatible PLE implementation; "
                "disabling PLE would change model outputs."
            )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        def get_layer(prefix: str):
            return Qwen4ExpDecoderLayer(
                vllm_config,
                config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            config,
            vllm_config.quant_config,
            f"{prefix}.hyper_connection_mixer",
            use_combine=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hc_count * config.hidden_size
        )
        self.norm = (
            Qwen4ExpRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states, positions)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.hyper_connection_mixer(hidden_states)
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Published Qwen4-Exp checkpoints include an MTP branch; the target
        # model is loaded independently and MTP weights must not be consumed by
        # the backbone until the dedicated hybrid MTP runner is implemented.
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
):
    hf_to_vllm_mapper = Qwen4ExpModel.hf_to_vllm_mapper

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        super().__init__()
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = (
            ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def set_moe_parameters(self):
        """Register MoE metadata without relying on Qwen3Next layer classes."""
        moe_blocks = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock)
        ]
        if not moe_blocks:
            raise RuntimeError("No Qwen4-Exp MoE layers found in the model.")
        self.moe_layers = [block.experts for block in moe_blocks]
        example = moe_blocks[0]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example.n_logical_experts
        self.num_physical_experts = example.n_physical_experts
        self.num_local_physical_experts = example.n_local_physical_experts
        self.num_routed_experts = example.n_routed_experts
        self.num_redundant_experts = example.n_redundant_experts

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return (2, len(self.model.layers) // 2, len(self.model.layers) - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The nested Qwen4ExpModel applies the HF-to-vLLM mapper. Applying it
        # here as well would rewrite already-packed names a second time.
        return AutoWeightsLoader(self, skip_prefixes=["mtp."]).load_weights(weights)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Multimodal wrapper; the published target currently requires PLE."""

    supports_multimodal = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            self.multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_mm_mapping(self):
        from vllm.model_executor.models.utils import MultiModelKeys

        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model="visual.",
        )


__all__ = [
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
]
