# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
"""Configuration classes for the Qwen4-Exp (Qwen3.8 Flash-Next) family.

The upstream Transformers implementation uses the same configuration names.
Keeping the classes in vLLM lets model/config discovery work with the pinned
Transformers version used by the SM75 runtime, before the model implementation
is available in a released Transformers wheel.
"""

from transformers.configuration_utils import PretrainedConfig


class Qwen4ExpTextConfig(PretrainedConfig):
    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    # This is also consumed by Transformers' native TP implementation.  The
    # vLLM model uses the same sharding contract for its native layers.
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
        "layers.*.self_attn.indexer.index_qk_proj": "colwise_gather_output",
        "layers.*.attn_hyper_connection.input_mix_weight_down": "rowwise_split_input",
        "layers.*.mlp_hyper_connection.input_mix_weight_down": "rowwise_split_input",
        "hyper_connection_mixer.input_mix_weight_down": "rowwise_split_input",
        "layers.*.ple.ple_embedding.ngram_embedding": "colwise_gather_output",
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=10,
        num_experts=512,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        full_attention_interval=4,
        hc_count=4,
        hc_lowrank=320,
        ple_layer_ids=None,
        ple_embed_dim=None,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
        split_ngram_parts=512,
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
        output_gate_type="silu",
        norm_topk_prob=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = sorted(set(ple_layer_ids or []))
        self.ple_embed_dim = hidden_size if ple_embed_dim is None else ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.seed = seed
        self.split_ngram_parts = split_ngram_parts
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.output_gate_type = output_gate_type or "silu"
        self.norm_topk_prob = norm_topk_prob

        if full_attention_interval <= 0:
            raise ValueError(
                "Qwen4-Exp full_attention_interval must be positive, got "
                f"{full_attention_interval}"
            )
        if layer_types is None:
            layer_types = [
                "linear_attention"
                if (i + 1) % full_attention_interval
                else "full_attention"
                for i in range(num_hidden_layers)
            ]
        elif len(layer_types) != num_hidden_layers:
            raise ValueError(
                "Qwen4-Exp layer_types must contain one entry per layer: "
                f"expected {num_hidden_layers}, got {len(layer_types)}"
            )
        # Keep ``full_attention`` in the public config: Transformers 5.10's
        # strict validator does not yet know the newer ``qwen_sparse_attention``
        # spelling.  The model layer factory treats full_attention as QSA.
        self.layer_types = list(layer_types)
        self.full_attention_interval = full_attention_interval
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        kwargs.setdefault("partial_rotary_factor", 0.25)
        kwargs.setdefault(
            "ignore_keys_at_rope_validation", {"mrope_section", "mrope_interleaved"}
        )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def validate_layer_type(self):
        unsupported = sorted(
            set(self.layer_types)
            - {"linear_attention", "full_attention", "qwen_sparse_attention"}
        )
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")

    def validate_architecture(self):
        self.validate_layer_type()
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}")
        if self.num_experts <= 0 or not (
            0 < self.num_experts_per_tok <= self.num_experts
        ):
            raise ValueError("Invalid Qwen4-Exp MoE expert counts")
        output_gate_type = self.output_gate_type or self.hidden_act
        if output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError(
                f"Unsupported Qwen4-Exp output gate activation: {output_gate_type}"
            )
        qsa = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        )
        if any(x is not None for x in qsa) and any(x is None for x in qsa):
            raise ValueError("Qwen4-Exp QSA requires all indexer fields")
        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            if self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads:
                raise ValueError(
                    "ple_embed_dim must be divisible by the PLE n-gram head count"
                )
            invalid = [
                i for i in self.ple_layer_ids if i < 1 or i > self.num_hidden_layers
            ]
            if invalid:
                raise ValueError(f"Invalid one-indexed PLE layer ids: {invalid}")


class Qwen4ExpVisionConfig(PretrainedConfig):
    model_type = "qwen4_exp_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen4ExpConfig(PretrainedConfig):
    model_type = "qwen4_exp"
    base_config_key = None
    sub_configs = {
        "vision_config": Qwen4ExpVisionConfig,
        "text_config": Qwen4ExpTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.text_config = (
            text_config
            if isinstance(text_config, Qwen4ExpTextConfig)
            else Qwen4ExpTextConfig(**(text_config or {}))
        )
        if isinstance(vision_config, Qwen4ExpVisionConfig):
            self.vision_config = vision_config
        else:
            vision_config = dict(vision_config or {})
            # Early Qwen4-Exp exports incorrectly used the top-level model type
            # for the vision sub-config.  Do not let that route back through the
            # text configuration registry.
            vision_config.pop("model_type", None)
            self.vision_config = Qwen4ExpVisionConfig(**vision_config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Qwen4ExpConfig", "Qwen4ExpTextConfig", "Qwen4ExpVisionConfig"]
