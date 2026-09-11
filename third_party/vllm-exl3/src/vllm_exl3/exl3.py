"""Portions of this module derive from Mia's AI Lab, overlay/exl3.py in
GLM-5.3-Flash-EXL3-2x-DGX-Sparks, first published 2026-08-27, which precedes this
project. The routed-expert EXL3/MCG path, its pointer-table construction, expert-map
pinning and diagnostic strings originate there.

Copyright (c) 2026 Mia's AI Lab. MIT. See THIRD_PARTY_NOTICES.md.

The EXL3 trellis format, the MCG codebook and the quantization method are ExLlamaV3's
work, Copyright (c) 2025 Turboderp, MIT. See THIRD_PARTY_NOTICES.md.
"""

# SPDX-License-Identifier: Apache-2.0
# EXL3 trellis quantization for routed experts, dense linears
# (non_routed_exl3), lm_head (ParallelLMHead) and row-wise n-gram embedding
# tables (ngram_embedding).
#
# Codebooks are mcg or mul1. Per-tensor K comes from layer_bits and
# non_routed_exl3.layers; matrices are padded to multiples of 128.
# Non-routed tensors without a spec stay native (UnquantizedLinearMethod).
#
# Experts never expand to a persistent BF16 weight; LinearEXL3 /
# exllamav3_ext runs the trellis GEMM. TP=2 shards gate/up column-wise and
# down row-wise; the MoE runner all-reduces the combined output.

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import re
try:
    import torch
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Parameter = None  # type: ignore[assignment]

try:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.linear import (
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.utils import set_weight_attrs
    _VLLM_AVAILABLE = True
    # Under the "vllm." hierarchy so vLLM's logging config actually emits these
    # INFO lines; a bare module name is dropped and the load log shows nothing.
    logger = init_logger("vllm." + __name__)
except ImportError:
    import logging
    _VLLM_AVAILABLE = False
    logger = logging.getLogger("vllm." + __name__)

    class FusedMoEQuantConfig:  # type: ignore[no-redef]
        pass

    class FusedMoEMethodBase:  # type: ignore[no-redef]
        pass

    class LinearBase:  # type: ignore[no-redef]
        pass

    class LinearMethodBase:  # type: ignore[no-redef]
        pass

    class UnquantizedLinearMethod:  # type: ignore[no-redef]
        pass

    class QuantizationConfig:  # type: ignore[no-redef]
        pass

    class QuantizeMethodBase:  # type: ignore[no-redef]
        pass

    def register_quantization_config(name: str):  # type: ignore[no-redef]
        def decorator(cls):
            return cls
        return decorator

    def set_weight_attrs(param, attrs):  # type: ignore[no-redef]
        for k, v in attrs.items():
            setattr(param, k, v)

MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
MUL1_MULTIPLIER = 0x83DCD12D
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
SWIGLU_LIMIT_DEFAULT = 10.0
TEMP_ROWS_FUSED = 2048
try:
    FAT_EXPERT_THRESHOLD = max(0, int(os.environ.get("VLLM_EXL3_FAT_THRESHOLD", "256")))
except (TypeError, ValueError):
    FAT_EXPERT_THRESHOLD = 256
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

# The schedule is deliberately expressed as inclusive ranges.  Keeping the
# policy here (rather than in a serving script) lets vLLM callers use the same
# batch-adaptive behaviour regardless of how the plugin is loaded.
DEFAULT_SPECULATIVE_SCHEDULE: list[tuple[int, int, int]] = [
    (1, 4, 3),
    (5, 8, 2),
    (9, 16, 1),
]
SPECULATIVE_SCHEDULE_ENV = "VLLM_EXL3_SPEC_SCHEDULE"
ADAPTIVE_VERIFICATION_ENV = "VLLM_EXL3_ADAPTIVE_VERIFICATION"


def compute_mla_kv_cache_bytes(
    context_len: int,
    num_layers: int = 43,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    dtype_bytes: int = 1,
) -> int:
    """Return the FP8 MLA KV-cache footprint for ``context_len`` tokens.

    DeepSeek-V4 stores one compressed KV latent and one decoupled RoPE key per
    layer.  The calculation is intentionally integer-only so callers can use
    it for an exact allocation or admission decision before starting a boot.
    """
    values = {
        "context_len": context_len,
        "num_layers": num_layers,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "dtype_bytes": dtype_bytes,
    }
    normalized: dict[str, int] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if integer != value:
            raise ValueError(f"{name} must be an integer")
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        normalized[name] = integer

    return (
        normalized["context_len"]
        * normalized["num_layers"]
        * (normalized["kv_lora_rank"] + normalized["qk_rope_head_dim"])
        * normalized["dtype_bytes"]
    )


def _env_float_override(default: float, *names: str, minimum: float | None = None) -> float:
    """Read the first valid finite float from a list of environment names."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip())
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or (minimum is not None and value < minimum):
            continue
        return value
    return default


def _env_int_override(default: int, *names: str, minimum: int | None = None) -> int:
    """Read the first valid integer from a list of environment names."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw.strip(), 10)
        except (AttributeError, TypeError, ValueError):
            continue
        if minimum is not None and value < minimum:
            continue
        return value
    return default


def validate_context_scaling(
    max_model_len: int,
    model_weights_gb: float = 95.4,
    total_mem_gb: float = 128.0,
    mem_util: float = 0.90,
    chunk_size: int = 2048,
) -> dict[str, float | int | bool]:
    """Validate an MLA context ceiling against a unified-memory budget.

    ``VLLM_EXL3_CONTEXT_*`` variables are the canonical overrides.  Shorter
    ``VLLM_EXL3_*`` aliases are accepted for shell compatibility.  Invalid
    values are ignored and leave the corresponding function argument intact.
    ``chunk_size`` is reported so callers can associate the result with their
    chunked-prefill configuration; it does not alter the static KV footprint.
    """
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise ValueError("max_model_len must be a positive integer")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    model_weights_gb = _env_float_override(
        float(model_weights_gb),
        "VLLM_EXL3_CONTEXT_MODEL_WEIGHTS_GB",
        "VLLM_EXL3_MODEL_WEIGHTS_GB",
    )
    total_mem_gb = _env_float_override(
        float(total_mem_gb),
        "VLLM_EXL3_CONTEXT_TOTAL_MEM_GB",
        "VLLM_EXL3_TOTAL_MEM_GB",
    )
    mem_util = _env_float_override(
        float(mem_util),
        "VLLM_EXL3_CONTEXT_MEM_UTIL",
        "VLLM_EXL3_MEM_UTIL",
    )
    chunk_size = _env_int_override(
        int(chunk_size),
        "VLLM_EXL3_CONTEXT_CHUNK_SIZE",
        "VLLM_EXL3_CHUNK_SIZE",
        minimum=1,
    )

    if not math.isfinite(model_weights_gb) or model_weights_gb < 0:
        raise ValueError("model_weights_gb must be finite and non-negative")
    if not math.isfinite(total_mem_gb) or total_mem_gb <= 0:
        raise ValueError("total_mem_gb must be finite and positive")
    if not math.isfinite(mem_util) or not 0.0 < mem_util <= 1.0:
        raise ValueError("mem_util must be finite and in (0.0, 1.0]")

    kv_cache_bytes = compute_mla_kv_cache_bytes(max_model_len)
    kv_cache_gb = kv_cache_bytes / (1024**3)
    usable_mem_gb = total_mem_gb * mem_util
    available_headroom_gb = usable_mem_gb - model_weights_gb - kv_cache_gb
    safety_margin_gb = total_mem_gb - model_weights_gb - kv_cache_gb
    return {
        "max_model_len": max_model_len,
        "kv_cache_bytes": kv_cache_bytes,
        "kv_cache_gb": kv_cache_gb,
        "usable_mem_gb": usable_mem_gb,
        "available_headroom_gb": available_headroom_gb,
        "fits": available_headroom_gb > 0.0,
        "safety_margin_gb": safety_margin_gb,
        "chunk_size": chunk_size,
    }


def _validated_speculative_schedule(schedule: object) -> list[tuple[int, int, int]] | None:
    """Return a normalized schedule, or ``None`` when it is invalid.

    A schedule with overlapping ranges is ambiguous, so it is rejected rather
    than silently depending on item order.  Gaps are valid and intentionally
    return zero draft tokens for the uncovered batch sizes.
    """
    if not isinstance(schedule, (list, tuple)) or not schedule:
        return None

    normalized: list[tuple[int, int, int]] = []
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        values: list[int] = []
        for value in entry:
            # Do not silently truncate floats (or accept booleans, which are
            # ``int`` subclasses) in a serving policy supplied by a caller.
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                values.append(value)
                continue
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
                values.append(int(value.strip(), 10))
                continue
            return None
        min_bs, max_bs, draft_tokens = values
        if min_bs < 1 or max_bs < min_bs or draft_tokens < 0:
            return None
        normalized.append((min_bs, max_bs, draft_tokens))

    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] <= previous[1]:
            return None
    return normalized


def parse_speculative_schedule(schedule_str: str) -> list[tuple[int, int, int]]:
    """Parse ``min_batch:max_batch:draft_tokens`` schedule entries.

    Invalid, empty, or ambiguous values safely fall back to a fresh copy of
    :data:`DEFAULT_SPECULATIVE_SCHEDULE`.  Returning a copy prevents a caller
    from mutating the process-wide default policy accidentally.
    """
    if not isinstance(schedule_str, str) or not schedule_str.strip():
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    entries: list[list[int]] = []
    try:
        for raw_entry in schedule_str.split(","):
            fields = [field.strip() for field in raw_entry.split(":")]
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError("each schedule entry must contain three integers")
            entries.append([int(field, 10) for field in fields])
    except (TypeError, ValueError, OverflowError):
        return list(DEFAULT_SPECULATIVE_SCHEDULE)

    normalized = _validated_speculative_schedule(entries)
    return normalized if normalized is not None else list(DEFAULT_SPECULATIVE_SCHEDULE)


def get_speculative_draft_tokens(
    batch_size: int,
    schedule: list | None = None,
) -> int:
    """Return the draft-token count for a scheduler batch size.

    ``schedule`` overrides :envvar:`VLLM_EXL3_SPEC_SCHEDULE`.  When neither is
    supplied, the default policy is 3/2/1 draft tokens for batches 1--4,
    5--8, and 9--16 respectively; all other batch sizes return zero.
    """
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError, OverflowError):
        return 0
    if batch_size <= 0:
        return 0

    if schedule is None:
        configured = os.environ.get(SPECULATIVE_SCHEDULE_ENV)
        active_schedule = (
            parse_speculative_schedule(configured)
            if configured is not None
            else list(DEFAULT_SPECULATIVE_SCHEDULE)
        )
    elif isinstance(schedule, str):
        active_schedule = parse_speculative_schedule(schedule)
    else:
        active_schedule = _validated_speculative_schedule(schedule)
        if active_schedule is None:
            active_schedule = list(DEFAULT_SPECULATIVE_SCHEDULE)

    for min_bs, max_bs, draft_tokens in active_schedule:
        if min_bs <= batch_size <= max_bs:
            return draft_tokens
    return 0


def is_adaptive_verification_enabled() -> bool:
    """Return whether confidence-based speculative verification is enabled.

    Only explicit affirmative values enable the feature.  This fail-closed
    policy keeps an unset, misspelled, or otherwise unknown environment value
    from changing verification behaviour unexpectedly in a serving process.
    """
    value = os.environ.get(ADAPTIVE_VERIFICATION_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def filter_speculative_candidates(
    probs: torch.Tensor,
    threshold: float = 0.5,
    *,
    return_tensor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | int]:
    """Keep the confident prefix of each speculative candidate sequence.

    Candidate verification is sequential: once a candidate falls below
    ``threshold``, that candidate and every later candidate in the same
    sequence are pruned.  A one-dimensional input is treated as one sequence
    and returns a Python ``int`` count; batched inputs return one ``long``
    count per leading sequence. ``return_tensor=True`` keeps a one-dimensional
    result's count as a scalar tensor instead of synchronizing to a Python int.
    """
    if isinstance(threshold, bool):
        raise ValueError("threshold must be finite and in [0.0, 1.0]")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be finite and in [0.0, 1.0]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0.0, 1.0]")

    if not isinstance(probs, torch.Tensor):
        raise TypeError(f"probs must be a torch.Tensor, got {type(probs).__name__}")

    if probs.ndim == 0:
        raise ValueError("probs must have a candidate dimension (at least 1D)")

    num_candidates = probs.shape[-1]
    if num_candidates == 0:
        mask = torch.zeros_like(probs, dtype=torch.bool)
        if probs.ndim == 1:
            count = torch.zeros((), dtype=torch.long, device=probs.device)
            return mask, count if return_tensor else 0
        return mask, torch.zeros(probs.shape[:-1], dtype=torch.long, device=probs.device)

    confident = torch.ge(probs, threshold)
    # cumprod encodes the first-failure cutoff without Python loops or host
    # synchronization, so the operation remains on the candidate tensor's
    # device during decode.
    mask = torch.cumprod(confident.to(dtype=torch.int64), dim=-1).to(dtype=torch.bool)
    kept_counts = mask.sum(dim=-1, dtype=torch.long)
    if probs.ndim == 1:
        return mask, kept_counts if return_tensor else int(kept_counts.item())
    return mask, kept_counts


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def _resolve_tp_geometry(*owners: Any) -> tuple[int, int]:
    """Resolve per-layer TP metadata before consulting process-wide TP state."""
    for owner in owners:
        if owner is None:
            continue
        # RoutedExperts keeps its effective geometry under ``moe_config``.
        # This is essential for EP: the process belongs to TP=4, while a
        # routed expert is deliberately whole (MoE TP=1) and assigned to an
        # EP rank.  Reading the process-wide group in that case would slice a
        # 640-wide EXL3 expert into invalid 160-wide packed fragments.
        moe_config = getattr(owner, "moe_config", None)
        if moe_config is not None:
            rank = getattr(moe_config, "tp_rank", None)
            size = getattr(moe_config, "tp_size", None)
            if rank is not None or size is not None:
                return (
                    int(rank) if rank is not None else 0,
                    int(size) if size is not None else 1,
                )
        rank = getattr(owner, "tp_rank", None)
        size = getattr(owner, "moe_tp_size", None)
        if size is None:
            size = getattr(owner, "tp_size", None)
        if size is None:
            size = getattr(owner, "_exl3_tp_size", None)
        if rank is not None or size is not None:
            return int(rank) if rank is not None else 0, int(size) if size is not None else 1

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Validate that the native ExLlamaV3 package and extension are importable."""
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    # ExLlamaV3 1.4+ imports cleanly as a regular package and its LinearEXL3
    # constructor relies on the real NullConfig/InferParams implementation.
    # Namespace stubs used by much older builds hide those classes and fail only
    # after a full checkpoint load, so deliberately exercise the normal import.
    importlib.import_module("exllamav3.modules.quant.exl3")


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    if out_dtype is None and torch is not None:
        out_dtype = torch.float16
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous() if mcg is not None else None,
        mul1=mul1.contiguous() if mul1 is not None else None,
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    if out_dtype is None and torch is not None:
        out_dtype = torch.float32
    inner = make_linear_exl3(
        trellis, suh, svh, mcg, mul1,
        out_dtype=torch.float16 if torch is not None else None,
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exllamav3_moe_available() -> bool:
    try:
        return callable(getattr(load_exllamav3_ext(), "exl3_moe", None))
    except Exception:
        return False


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    raw_id = id(emap)
    cached = getattr(layer, "_exl3_pinned_expert_map", None)
    if (
        getattr(layer, "_exl3_raw_expert_map_id", None) == raw_id
        and cached is not None
        and cached.device == device
        and cached.dtype == torch.long
    ):
        return cached
    pinned = emap.to(device=device, dtype=torch.long)
    layer._exl3_pinned_expert_map = pinned
    layer._exl3_raw_expert_map_id = raw_id
    return pinned


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        if limit is not None and limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        act = F.silu(gate) * up
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def apply_exl3_reconstruct_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor:
    """Reject expert slices that the EXL3 extension cannot execute correctly."""
    del x2d, ids, weights, inners, expert_map, limit
    raise RuntimeError(
        "EXL3 routed experts require a 128-aligned local intermediate width; "
        "Qwen Flash TP=4 creates 160-wide slices. Use --enable-expert-parallel "
        "with TP=4 so every rank owns complete 640-wide experts."
    )


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    try:
        exllamav3_ext = load_exllamav3_ext()
    except Exception:
        exllamav3_ext = None

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    # Short aliases match the native extension terminology and keep the table
    # ABI stable for callers that construct their own RoutedExperts wrapper.
    layer._exl3_ptrs.update(
        {
            "gate_t_ptrs": layer._exl3_ptrs["gate_trellis"],
            "gate_suh_ptrs": layer._exl3_ptrs["gate_suh"],
            "gate_svh_ptrs": layer._exl3_ptrs["gate_svh"],
            "up_t_ptrs": layer._exl3_ptrs["up_trellis"],
            "up_suh_ptrs": layer._exl3_ptrs["up_suh"],
            "up_svh_ptrs": layer._exl3_ptrs["up_svh"],
            "down_t_ptrs": layer._exl3_ptrs["down_trellis"],
            "down_suh_ptrs": layer._exl3_ptrs["down_suh"],
            "down_svh_ptrs": layer._exl3_ptrs["down_svh"],
        }
    )
    idx = int(device.index) if device.index is not None else 0
    if exllamav3_ext is not None and hasattr(
        exllamav3_ext, "exl3_moe_max_concurrency"
    ):
        concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
        if concurrency < 1:
            concurrency = 1
    else:
        layer._exl3_fused_temps = None
        layer._exl3_fused_concurrency = 0
        layer._exl3_k = int(layer._exl3_bits)
        return
    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, hidden), dtype=torch.float32, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, TEMP_ROWS_FUSED, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)


_FAT_SCRATCH_CACHE: dict[tuple[str, int, int, int, int], dict[str, torch.Tensor]] = {}


def _fat_scratch(
    device: torch.device, capacity: int, gate: Any
) -> dict[str, torch.Tensor]:
    hidden = int(getattr(gate, "in_features", 4096))
    intermediate = int(getattr(gate, "out_features", 2048))
    k_words = int(gate.trellis.shape[2])
    bucketed_cap = max(256, ((int(capacity) + 255) // 256) * 256)
    key = (str(device), bucketed_cap, intermediate, hidden, k_words)
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (bucketed_cap, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (bucketed_cap, hidden), dtype=torch.float32, device=device
        ),
        "w_gate": torch.empty(
            (hidden, intermediate), dtype=torch.float16, device=device
        ),
        "w_up": torch.empty(
            (hidden, intermediate), dtype=torch.float16, device=device
        ),
        # Contiguous per-projection outputs for the distinct-suh branch: the
        # extension GEMM and Hadamard kernels index row-major contiguous
        # operands, so column slices of ``gate_up`` must not be handed to them.
        "g_tmp": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
        "u_tmp": torch.empty(
            (bucketed_cap, intermediate), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    return scratch


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float | None,
    cap: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run fat experts with persistent ExLlamaV3 reconstruction scratch."""
    ext = load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        distinct_suh = not torch.equal(gate.suh, up.suh)
        if not distinct_suh:
            ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)

        k = int(getattr(gate, "K", 4))
        mcg = bool(getattr(gate, "mcg", True))
        mul1 = bool(getattr(gate, "mul1", False))

        if distinct_suh:
            gate_h = h13
            up_h = h
            ext.had_r_128(h, gate_h, gate.suh, None, 1.0)
            ext.had_r_128(up_h, up_h, up.suh, None, 1.0)
            w_gate = scratch["w_gate"]
            w_up = scratch["w_up"]
            ext.reconstruct(w_gate, gate.trellis, k, mcg, mul1)
            ext.reconstruct(w_up, up.trellis, k, mcg, mul1)
            g_tmp = scratch["g_tmp"][:n_rows]
            u_tmp = scratch["u_tmp"][:n_rows]
            ext.hgemm(gate_h, w_gate, g_tmp)
            ext.hgemm(up_h, w_up, u_tmp)
            ext.had_r_128(g_tmp, g_tmp, None, gate.svh, 1.0)
            ext.had_r_128(u_tmp, u_tmp, None, up.svh, 1.0)
        else:
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, k, mcg, mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        if distinct_suh:
            gate_out = scratch["g_tmp"][:n_rows]
            up_out = scratch["u_tmp"][:n_rows]
        else:
            gate_out = gate_up[:, :intermediate]
            up_out = gate_up[:, intermediate:]
        if limit is not None and limit > 0:
            gate_out.clamp_(max=limit)
            up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        w2 = scratch["w2"]
        ext.reconstruct(w2, down.trellis, k, mcg, mul1)
        down_out = scratch["down"][:n_rows]
        ext.hgemm(h2, w2, down_out)
        ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
        down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
        out.index_add_(0, token_idx, down_out)
    return out


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float | None = None,
) -> torch.Tensor:
    """One exl3_moe launch per layer. Experts with count > 128 fall back to LinearEXL3."""
    tokens, hidden = x2d.shape
    n_exp = len(inners)

    try:
        import exllamav3_ext
    except Exception:
        # Direct callers may use this helper without installing ExLlamaV3.
        # Keep the same graceful fallback contract as ``apply_exl3_experts``.
        return apply_exl3_python_loop(
            x2d, ids, weights, inners, expert_map, limit
        )

    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]

    if tokens > TEMP_ROWS_FUSED and bool((counts > TEMP_ROWS_FUSED).any().item()):
        logger.info_once("EXL3 fat-chunk slicing ACTIVE (tokens=%d)" % tokens)
        # Deep-context prefill chunks can route more than TEMP_ROWS_FUSED rows
        # to a single expert. The fused kernel covers at most TEMP_ROWS_FUSED
        # rows per expert, and the old fallback reconstructed whole experts
        # per chunk, stalling prefill by orders of magnitude past ~160k
        # context (the ">163k hang"). Within a slice of <= TEMP_ROWS_FUSED
        # tokens no expert can exceed TEMP_ROWS_FUSED rows (each token adds at
        # most one row per expert), so re-run the fused path per slice.
        # Prefill-only: decode batches are at most the largest capture size,
        # far below TEMP_ROWS_FUSED, and never reach this host sync.
        for s in range(0, tokens, TEMP_ROWS_FUSED):
            e = min(s + TEMP_ROWS_FUSED, tokens)
            out[s:e] = apply_exl3_fused_moe(
                x2d[s:e], ids[s:e], weights[s:e], layer, inners, expert_map, limit
            )
        return out

    fat = counts > FAT_EXPERT_THRESHOLD
    # counts[e] cannot exceed the row count, so with no more rows than the
    # threshold no expert can be fat. Testing that Python-side first keeps the
    # device sync below off the decode path, where graph capture forbids it.
    fat_possible = tokens > FAT_EXPERT_THRESHOLD
    fat_route = torch.zeros_like(local, dtype=torch.bool)
    if fat_possible and bool(fat.any().item()):
        safe_local = local.clamp(min=0, max=max(n_exp - 1, 0))
        fat_route = (local < n_exp) & fat.index_select(0, safe_local)

    # The standard kernel handles non-fat routes. Fat routes are represented by
    # the invalid sentinel with zero weight here and are dispatched exactly once
    # below through the fat GEMM path.
    standard_local = local.masked_fill(fat_route, n_exp)
    standard_weight = flat_weight.masked_fill(fat_route, 0)
    order = standard_local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = standard_weight[order]
    standard_count = torch.zeros(
        n_exp + 1, dtype=torch.long, device=local.device
    )
    standard_count.scatter_add_(
        0,
        standard_local.long(),
        torch.ones(standard_local.shape, dtype=torch.long, device=local.device),
    )
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None

    k = int(getattr(layer, "_exl3_k", 4))
    args = (
        xh,
        out,
        standard_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        temps[4],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        *getattr(layer, "_exl3_codebook_flags", (True, False, True, False, True, False)),
        float(limit) if (limit is not None and limit > 0) else 0.0,
    )
    if n_active_host is not None:
        fn(*args, n_active_host, False)
    else:
        fn(*args, False)

    if fat_possible and bool(fat.any().item()):
        fat_order = local.argsort()
        apply_exl3_batched_fat(
            xh,
            flat_token[fat_order],
            flat_weight[fat_order],
            counts.tolist(),
            inners,
            limit,
            FAT_EXPERT_THRESHOLD,
            out,
        )
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float | None = None,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    if _EXL3_PREFILL_SYNC:
        _prefill_sync(int(x.numel() // x.shape[-1]))
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)

    # ``exl3_moe`` assumes that the local intermediate width is a multiple of
    # 128. Qwen Flash's 640-wide expert MLP is 160-wide at TP=4, for which the
    # fused kernel silently omits the final tile. Use the eager reconstruct
    # fallback before selecting native/fused dispatch to preserve correctness.
    local_intermediate = int(getattr(layer, "_exl3_intermediate_local", 0))
    if local_intermediate and local_intermediate % 128:
        out = apply_exl3_reconstruct_moe(
            x2d, ids, weights, inners, expert_map, limit
        )
        layer._exl3_last_apply = "reconstruct"
        return out.to(dtype=x.dtype)

    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _exl3_pad128(n: int) -> int:
    """EXL3 stores matrices padded to multiples of 128 on both dims."""
    return (int(n) + 127) // 128 * 128


def _prefix_has_suffix(prefix: str, suffix: str) -> bool:
    """Module-path suffix match: "self_attn.o_proj" matches
    "model.layers.3.self_attn.o_proj" but not "...cross_attn.o_proj_x"."""
    return prefix == suffix or prefix.endswith("." + suffix)


def _native_storage_vllm_prefix(prefix: str) -> str:
    """Map Qwen3.5 native-pack names onto vLLM's constructed modules."""
    if prefix == "lm_head":
        return "language_model.lm_head"
    if prefix.startswith("model.language_model."):
        prefix = prefix.replace(
            "model.language_model.", "language_model.model.", 1
        )

    for source, destination in (
        (".linear_attn.in_proj_qkv", ".linear_attn.in_proj_qkvz"),
        (".linear_attn.in_proj_z", ".linear_attn.in_proj_qkvz"),
        (".self_attn.q_proj", ".self_attn.qkv_proj"),
        (".self_attn.k_proj", ".self_attn.qkv_proj"),
        (".self_attn.v_proj", ".self_attn.qkv_proj"),
        (".mlp.gate_proj", ".mlp.gate_up_proj"),
        (".mlp.up_proj", ".mlp.gate_up_proj"),
    ):
        if prefix.endswith(source):
            return f"{prefix[: -len(source)]}{destination}"
    return prefix


def _native_storage_exl3_layers(
    tensor_storage: dict[str, Any] | None,
) -> dict[str, int]:
    """Extract vLLM dense-linear bit widths from an ExLlamaV3 native pack."""
    layers: dict[str, int] = {}
    for source_prefix, entry in (tensor_storage or {}).items():
        if not isinstance(entry, dict) or entry.get("quant_format") != "exl3":
            continue
        bits = int(entry.get("bits_per_weight", 0))
        if bits not in (2, 3, 4, 5, 6):
            raise ValueError(
                f"unsupported EXL3 bits={bits} for native tensor {source_prefix}"
            )
        prefix = _native_storage_vllm_prefix(source_prefix)
        previous = layers.setdefault(prefix, bits)
        if previous != bits:
            raise ValueError(
                "native EXL3 tensors merged into one vLLM layer must use the "
                f"same bit width: {prefix} has {previous} and {bits}"
            )
    return layers


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        self.scope = str(scope)
        # Optional per-layer override, e.g. {"42": 3, "27": 3}. Layers absent
        # from the map use `bits`. This is how a mixed-K checkpoint (K2 base
        # with K3 delta layers) declares itself; the trellis tensors for those
        # layers are shaped for their own K and would fail the load shape
        # check under the base K.
        raw_layer_bits = kwargs.pop("layer_bits", None) or {}
        self.layer_bits: dict[int, int] = {
            int(k): int(v) for k, v in dict(raw_layer_bits).items()
        }
        for layer_idx, layer_k in self.layer_bits.items():
            if layer_k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported EXL3 bits={layer_k} for layer {layer_idx}"
                )
        # Non-routed dense linear config: optional {"modules": [...], "bits": K, "layer_bits": {...},
        # "codebook": "mcg"|"mul1", "layers": {prefix: {"bits": K, "bf16_shards": [...]}, ...}}
        raw_nr_exl3 = kwargs.pop("non_routed_exl3", None) or {}
        self.non_routed_exl3: dict[str, Any] = dict(raw_nr_exl3) if raw_nr_exl3 else {}
        self.native_storage_exl3_layers: dict[str, int] = {}
        mtp_bits = kwargs.pop("mtp_bits", None)
        self.mtp_bits = int(mtp_bits) if mtp_bits is not None else None
        if self.mtp_bits is not None and self.mtp_bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 MTP bits={self.mtp_bits}")
        head_bits = kwargs.pop("head_bits", None)
        self.head_bits = int(head_bits) if head_bits is not None else None
        if self.head_bits is not None and self.head_bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 head bits={self.head_bits}")
        # Validate non-routed bits if present
        nr_bits = self.non_routed_exl3.get("bits")
        if nr_bits is not None and nr_bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported non_routed_exl3 bits={nr_bits}")
        nr_layer_bits = self.non_routed_exl3.get("layer_bits", {})
        for suffix, k in (nr_layer_bits or {}).items():
            if k not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported non_routed_exl3 bits={k} for suffix {suffix}"
                )
        # Validate non-routed layers dict: each value is {"bits": K[, "bf16_shards": [...]]}
        nr_layers = self.non_routed_exl3.get("layers", {})
        for prefix, layer_cfg in (nr_layers or {}).items():
            if not isinstance(layer_cfg, dict):
                raise ValueError(
                    f"non_routed_exl3 layers[{prefix}] must be a dict, got {type(layer_cfg)}"
                )
            layer_bits = layer_cfg.get("bits")
            if layer_bits is not None and layer_bits not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported non_routed_exl3 layers[{prefix}] bits={layer_bits}"
                )
        # Validate non-routed codebook
        nr_codebook = self.non_routed_exl3.get("codebook", "mcg")
        if nr_codebook not in ("mcg", "mul1"):
            raise ValueError(
                f"unsupported non_routed_exl3 codebook={nr_codebook!r}; must be 'mcg' or 'mul1'"
            )
        # Optional row-wise trellis embedding table in exllamav3's n-gram format
        # (Qwen3.8-Flash-Next PLE), e.g. {"bits": 5, "num_shards": 128,
        # "rows_per_shard": 2500012, "num_heads": 16, "modules": ["ngram_embedding"]}.
        raw_ngram = kwargs.pop("ngram_embedding", None) or {}
        self.ngram_embedding: dict[str, Any] = dict(raw_ngram) if raw_ngram else {}
        if self.ngram_embedding:
            for key in ("bits", "num_shards", "rows_per_shard", "num_heads"):
                if int(self.ngram_embedding.get(key, 0) or 0) <= 0:
                    raise ValueError(f"ngram_embedding.{key} must be a positive integer")
            if int(self.ngram_embedding["bits"]) not in range(1, 9):
                raise ValueError(
                    f"unsupported ngram_embedding bits={self.ngram_embedding['bits']}"
                )
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mcg", "mul1"):
            raise ValueError(
                f"unsupported codebook={self.codebook!r}; must be 'mcg' or 'mul1'"
            )
        if self.bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")


    def _mtp_expert_method(self, layer, prefix):
        """Quant method for draft/MTP experts, which stay in the base format.

        These are the model's own experts, fp4 for DSV4: packed weights with
        E8M0 block scales, which vLLM loads by looking up w13_weight_scale. The
        non-routed delegate describes fp8 block quantization for the attention
        and dense layers and creates w13_weight_scale_inv instead, so it cannot
        serve these. Prefer MXFP4, whose MoE method creates the expected names,
        and keep the non-routed delegate as the fallback for packs that are not
        fp4. Returns (method, description).
        """
        if str(getattr(self, "mtp_expert_dtype", "fp4")) == "fp4":
            try:
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4Config,
                )

                method = Mxfp4Config().get_quant_method(layer, prefix)
                if method is not None:
                    return method, "mxfp4"
            except Exception as exc:  # pragma: no cover - depends on vLLM build
                if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                    print(f"[exl3-routing] mxfp4 delegate unusable: {exc}", flush=True)
        delegate = self._non_routed_delegate()
        if delegate is not None:
            method = delegate.get_quant_method(layer, prefix)
            if method is not None:
                return method, "non_routed"
        return None, "none"

    def get_name(self) -> str:
        return "exl3"

    _LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def bits_for_prefix(self, prefix: str) -> int:
        """Per-layer K: `layer_bits` entry for this layer, else the base K."""
        if not self.layer_bits:
            return self.bits
        m = self._LAYER_RE.search(prefix or "")
        if m is None:
            return self.bits
        return self.layer_bits.get(int(m.group(1)), self.bits)

    def _matches_non_routed_exl3(self, prefix: str) -> bool:
        """Check if prefix matches non_routed_exl3: either layers dict keys or modules list."""
        if prefix in self.native_storage_exl3_layers:
            return True
        if not self.non_routed_exl3:
            return False
        # Check if prefix is a key in the layers dict
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            return True
        # Fall back to suffix matching on modules list
        modules = self.non_routed_exl3.get("modules", [])
        if not modules:
            return False
        return any(_prefix_has_suffix(prefix, m) for m in modules)

    def _matches_mtp_exl3(self, prefix: str) -> bool:
        """Whether this is a quantized Qwen3.5 MTP linear projection."""
        # The draft model remaps ``mtp.*`` weights into its local ``model.*``
        # tree, while its borrowed output head remains named ``lm_head``.
        return self.mtp_bits is not None and prefix.startswith("mtp.")

    def _bits_for_non_routed(self, prefix: str) -> int:
        """Get K bits for non_routed_exl3 layer, checking layers dict first, then suffix form."""
        native_bits = self.native_storage_exl3_layers.get(prefix)
        if native_bits is not None:
            return native_bits
        if not self.non_routed_exl3:
            return self.bits
        # Check layers dict first
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            layer_cfg = layers[prefix]
            if "bits" in layer_cfg:
                return int(layer_cfg["bits"])
            return int(self.non_routed_exl3.get("bits", self.bits))
        # Fall back to suffix matching
        modules = self.non_routed_exl3.get("modules", [])
        matched_suffix = None
        for suffix in modules:
            if _prefix_has_suffix(prefix, suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            return self.bits
        # Check layer_bits override for this suffix
        layer_bits = self.non_routed_exl3.get("layer_bits", {})
        if matched_suffix in layer_bits:
            return int(layer_bits[matched_suffix])
        # Fall back to non_routed_exl3 bits or base bits
        return int(self.non_routed_exl3.get("bits", self.bits))

    def _bf16_shards_for(self, prefix: str) -> list[int]:
        """Get bf16 shard indices for a non_routed_exl3 layer from the layers dict."""
        if not self.non_routed_exl3:
            return []
        layers = self.non_routed_exl3.get("layers", {})
        if layers and prefix in layers:
            layer_cfg = layers[prefix]
            return list(layer_cfg.get("bf16_shards", []))
        return []

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # ExLlamaV3-Turing supplies the SM75 implementation used by this
        # experimental integration. The GB10-only extension stays optional.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "codebook",
            "scope",
            "quant_method",
            # Some packs ship a large per-tensor ledger here; keep it off the config object.
            "tensor_storage",
            "non_routed_exl3",
            "non_routed_quantization",
            "mtp_experts",
            "mtp_experts_start_layer",
        }
        inst = cls(
            bits=int(config.get("bits", 4)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            non_routed_exl3=config.get("non_routed_exl3"),
            **{k: v for k, v in config.items() if k not in skip},
        )
        # __init__ swallows unknown kwargs; stash the delegation dict explicitly.
        inst.non_routed_quantization = config.get("non_routed_quantization")
        inst.native_storage_exl3_layers = _native_storage_exl3_layers(
            config.get("tensor_storage")
        )
        # "bf16_as_stored": dense linears are BF16 tensors; never delegate them
        # (the delegate still serves source-format MTP experts).
        inst.non_routed_dtype_policy = str(config.get("non_routed_dtype_policy", ""))
        # Mixed-format packs: draft/MTP blocks appended past the main stack can
        # keep their experts in the source format (e.g. MXFP4). Declare
        # mtp_experts: "source" plus mtp_experts_start_layer: <first draft
        # layer index>; those layers delegate to non_routed_quantization.
        inst.mtp_experts = str(config.get("mtp_experts", "exl3"))
        inst.mtp_experts_start_layer = config.get("mtp_experts_start_layer")
        return inst

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def _ngram_embedding_spec(self, prefix: str) -> dict[str, Any] | None:
        """The n-gram table spec if ``prefix`` names one of its modules, else None."""
        spec = getattr(self, "ngram_embedding", None) or {}
        if not spec:
            return None
        modules = list(spec.get("modules") or ["ngram_embedding"])
        for m in modules:
            if prefix == m or prefix.endswith("." + m):
                return spec
        return None

    def disable_embedding_tensor_parallel(self, prefix: str) -> bool:
        """Keep a streamed hash table whole on each TP worker.

        PLE ids address a global hash table, rather than a vocabulary partition.
        ``VocabParallelEmbedding`` must therefore skip its mask/all-reduce path.
        """
        return self._ngram_embedding_spec(prefix) is not None and _ngram_stream_enabled()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            # Draft/MTP blocks construct with plain layers.N prefixes (the
            # mtp_block name appears only in parameter paths), so gate by
            # declared layer index, never by name.
            if getattr(self, "mtp_experts", "exl3") == "source":
                _start = getattr(self, "mtp_experts_start_layer", None)
                _lm = re.search(r"layers\.(\d+)\.", prefix)
                _by_index = bool(
                    _start is not None and _lm and int(_lm.group(1)) >= int(_start)
                )
                if _by_index:
                    dm, how = self._mtp_expert_method(layer, prefix)
                    if dm is not None:
                        if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                            print(
                                f"[exl3-routing] MoE {prefix} -> source via {how} "
                                f"(by_index={_by_index})",
                                flush=True,
                            )
                        return dm
            if os.environ.get("VLLM_EXL3_LOG_MOE_ROUTING"):
                print(f"[exl3-routing] MoE {prefix} -> exl3", flush=True)
            return Exl3MoEMethod(
                layer.moe_config, self, bits=self.bits_for_prefix(prefix)
            )
        if isinstance(layer, LinearBase):
            if self._matches_mtp_exl3(prefix):
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=self.mtp_bits)
            if prefix == "lm_head" and self.head_bits is not None:
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=self.head_bits)
            # Check if this LinearBase should use non_routed_exl3
            if self._matches_non_routed_exl3(prefix):
                bits = self._bits_for_non_routed(prefix)
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=bits)
            if getattr(self, "non_routed_dtype_policy", "") == "bf16_as_stored":
                return UnquantizedLinearMethod()
            d = self._non_routed_delegate()
            if d is not None:
                m = d.get_quant_method(layer, prefix)
                if m is not None:
                    return m
            return UnquantizedLinearMethod()
        # Embedding-family layers. vLLM only consults quant_config for these when
        # the model passes it (qwen4_exp needs quant_config= on ParallelLMHead and
        # on the PLE table). lm_head is an ordinary trellis linear; the PLE n-gram
        # table is the row-wise exllamav3 format served by Exl3EmbeddingMethod.
        try:
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
                VocabParallelEmbedding,
            )
        except ImportError:  # pragma: no cover
            return None
        if isinstance(layer, ParallelLMHead):
            if self._matches_non_routed_exl3(prefix):
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=self._bits_for_non_routed(prefix))
            if prefix == "lm_head" and self.head_bits is not None:
                layer._exl3_prefix = prefix
                return Exl3LinearMethod(self, bits=self.head_bits)
            return None
        if isinstance(layer, VocabParallelEmbedding):
            spec = self._ngram_embedding_spec(prefix)
            if spec is not None:
                layer._exl3_prefix = prefix
                return Exl3EmbeddingMethod(self, spec)
            return None
        return None

    def _non_routed_delegate(self):
        # Packs that keep non-routed weights in the official source format
        # (e.g. DeepSeek block-FP8) declare it under
        # ``quantization_config.non_routed_quantization``; delegate those
        # layers to the matching quant method so arch-specific fp8 forward
        # paths get real scale tensors. Absent key = unquantized (GLM).
        if not hasattr(self, "_nr_delegate_cached"):
            self._nr_delegate_cached = None
            nrq = getattr(self, "non_routed_quantization", None)
            if isinstance(nrq, dict) and nrq.get("quant_method"):
                from vllm.model_executor.layers.quantization import (
                    get_quantization_config,
                )
                name = str(nrq["quant_method"])
                try:
                    cls = get_quantization_config(name)
                except Exception as exc:
                    raise RuntimeError(
                        "Unable to load declared non_routed_quantization delegate "
                        f"quant_method={name!r} config={nrq!r}"
                    ) from exc
                if cls is None:
                    raise ValueError(
                        "Declared non_routed_quantization delegate is unavailable: "
                        f"quant_method={name!r} config={nrq!r}"
                    )
                try:
                    self._nr_delegate_cached = cls.from_config(dict(nrq))
                except Exception as exc:
                    raise ValueError(
                        "Invalid declared non_routed_quantization delegate "
                        f"quant_method={name!r} config={nrq!r}"
                    ) from exc
        return self._nr_delegate_cached


# Mirrors the checkpoint-name resolution of vLLM's RoutedExperts.load_weights
# (vllm-project/vllm, Apache-2.0); see THIRD_PARTY_NOTICES.md.
def _exl3_routed_experts_loader(layer: torch.nn.Module):
    """Per-expert ``load_weights`` for a RoutedExperts layer holding EXL3 tensors.

    Mirrors vLLM's ``RoutedExperts.load_weights`` name resolution but never takes its
    fused (3-D) branch: an EXL3 checkpoint always stores one tensor per expert.
    """

    def load_weights(weights):
        try:
            mapping = layer.get_expert_mapping(include_fused=True)
        except TypeError:
            mapping = layer.get_expert_mapping()
        layer_name = str(getattr(layer, "layer_name", ""))
        for expert_name, loaded_weight in weights:
            qual_name = f"{layer_name}.{expert_name}" if layer_name else expert_name
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in qual_name:
                    continue
                full_name = qual_name.replace(weight_name, param_name)
                local_name = full_name.removeprefix(f"{layer_name}.")
                param = getattr(layer, local_name, None)
                if param is None:
                    if local_name.endswith(("w13_bias", "w2_bias")):
                        break
                    raise AttributeError(
                        f"EXL3 routed experts {layer_name!r} has no parameter "
                        f"{local_name!r} for checkpoint weight {qual_name!r}"
                    )
                ok = param.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=full_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if ok:
                    yield local_name
                break

    return load_weights


def _moe_marker_or_none(marker: torch.Tensor):
    """A codebook marker tensor if it was loaded (non-zero), else None."""
    return marker if int(marker.reshape(-1)[0].item()) != 0 else None


def _check_moe_codebook_markers(mcg: torch.Tensor, mul1: torch.Tensor, what: str) -> None:
    """Every expert tensor carries exactly one codebook marker with the known value."""
    mcg_v = mcg.reshape(-1)
    mul1_v = mul1.reshape(-1)
    mcg_set = mcg_v != 0
    mul1_set = mul1_v != 0
    if bool((mcg_set & mul1_set).any()):
        raise RuntimeError(f"EXL3 {what}: an expert tensor has both mcg and mul1 markers")
    if bool((~mcg_set & ~mul1_set).any()):
        raise RuntimeError(
            f"EXL3 {what}: an expert tensor has no codebook marker (mcg or mul1 never loaded)"
        )
    if bool((mcg_v[mcg_set] != MCG_MARKER_SIGNED_INT32).any()):
        raise RuntimeError(
            f"EXL3 {what}: mcg marker is not the MCG int32 {MCG_MARKER_SIGNED_INT32}; "
            "packed ABI mismatch"
        )
    if bool((mul1_v[mul1_set] != MUL1_MARKER_SIGNED_INT32).any()):
        raise RuntimeError(
            f"EXL3 {what}: mul1 marker is not the mul1 int32 {MUL1_MARKER_SIGNED_INT32}; "
            "packed ABI mismatch"
        )


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed MCG trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(
        self, moe, quant_config: Exl3Config, bits: int | None = None
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        # One method instance per RoutedExperts layer, so this is per-layer K.
        self.bits = int(bits) if bits is not None else quant_config.bits
        self._logged = False

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # w13_* : stacked [expert, {gate=0, up=1}, ...] so the stock
        # expert_params_mapping (experts.w13_ + suffix) hits these names.
        w13_trellis = Parameter(
            torch.empty(
                num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(
                num_experts, 2, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_mcg = Parameter(
            torch.zeros(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w13_mul1 = Parameter(
            torch.zeros(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.empty(
                num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mcg = Parameter(
            torch.zeros(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_mul1 = Parameter(
            torch.zeros(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mcg": w13_mcg,
            "w13_mul1": w13_mul1,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mcg": w2_mcg,
            "w2_mul1": w2_mul1,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits
        # vLLM's generic RoutedExperts.load_weights treats any 3-D checkpoint
        # tensor as fused stacked experts and unbinds it per expert; an EXL3
        # per-expert trellis is 3-D by construction. Route this layer's tensors
        # through a per-expert loader instead (instance attribute shadows the
        # class method for AutoWeightsLoader; direct-calling models are unaffected).
        layer.load_weights = _exl3_routed_experts_loader(layer)

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank, tp_size = _resolve_tp_geometry(owner, layer)
        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()
        if suffix in ("mcg", "mul1"):
            # Codebook markers are scalars ([] or [1]); keep the value per expert
            # tensor so process_weights_after_loading can pick the codebook.
            if shard_id in ("w1", "w3"):
                dest = param.data[expert_id, 0 if shard_id == "w1" else 1]
            elif shard_id == "w2":
                dest = param.data[expert_id]
            else:
                raise ValueError(f"unknown EXL3 shard_id={shard_id}")
            dest.fill_(int(loaded.reshape(-1)[0].item()) if loaded.numel() else 0)
            return True if return_success else None
        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mcg",
            "w13_mul1",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mcg",
            "w2_mul1",
        ):
            getattr(layer, name)._exl3_owner = layer
        _check_moe_codebook_markers(layer.w13_mcg, layer.w13_mul1, "w13")
        _check_moe_codebook_markers(layer.w2_mcg, layer.w2_mul1, "w2")

        n_exp = int(layer.w13_trellis.shape[0])
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            gate = make_linear_exl3(
                layer.w13_trellis[e, 0],
                layer.w13_suh[e, 0],
                layer.w13_svh[e, 0],
                _moe_marker_or_none(layer.w13_mcg[e, 0]),
                _moe_marker_or_none(layer.w13_mul1[e, 0]),
            )
            up = make_linear_exl3(
                layer.w13_trellis[e, 1],
                layer.w13_suh[e, 1],
                layer.w13_svh[e, 1],
                _moe_marker_or_none(layer.w13_mcg[e, 1]),
                _moe_marker_or_none(layer.w13_mul1[e, 1]),
            )
            down = make_linear_exl3(
                layer.w2_trellis[e],
                layer.w2_suh[e],
                layer.w2_svh[e],
                _moe_marker_or_none(layer.w2_mcg[e]),
                _moe_marker_or_none(layer.w2_mul1[e]),
            )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners
        # Codebook flags (mcg, mul1) per projection for the fused kernel launch;
        # every expert in a layer must agree.
        if inners:
            flags = tuple(
                bool(getattr(inners[0][w], a, d))
                for w in ("gate", "up", "down")
                for a, d in (("mcg", True), ("mul1", False))
            )
            for e, inner in enumerate(inners):
                f_e = tuple(
                    bool(getattr(inner[w], a, d))
                    for w in ("gate", "up", "down")
                    for a, d in (("mcg", True), ("mul1", False))
                )
                if f_e != flags:
                    raise RuntimeError(
                        f"EXL3 experts disagree on codebook: expert {e} {f_e} vs expert 0 {flags}"
                    )
            layer._exl3_codebook_flags = flags
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                if _exllamav3_moe_available():
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "no ExLlamaV3 MoE kernel available"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if not self._logged and self.bits != self.quant_config.bits:
            logger.info(
                "EXL3 per-layer K override: layer prefix %s uses bits=%d (base %d)",
                getattr(layer, "layer_name", None) or getattr(layer, "prefix", "?"),
                self.bits,
                self.quant_config.bits,
            )
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        raw_limit = getattr(self.moe, "swiglu_limit", None)
        try:
            parsed_limit = float(raw_limit)
        except (TypeError, ValueError, OverflowError):
            parsed_limit = None
        limit = (
            parsed_limit
            if parsed_limit is not None
            and math.isfinite(parsed_limit)
            and parsed_limit > 0
            else None
        )
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=limit
        )


# ---------------------------------------------------------------------------
# Row-wise EXL3 embedding tables (exllamav3 n-gram format)
# ---------------------------------------------------------------------------

NGRAM_ROW_DIM = 160
NGRAM_MUL1 = 0x83DCD12D


def _ngram_stream_enabled() -> bool:
    raw = os.environ.get("VLLM_EXL3_NGRAM_STREAM", "0").strip().lower()
    if raw in ("0", "false", "no", "off", ""):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(
        "VLLM_EXL3_NGRAM_STREAM must be one of 0/1/false/true, "
        f"got {raw!r}"
    )


def ngram_words_per_row(bits: int) -> int:
    """Packed int16 words per row: one fp16 scale word plus the K-bit ring bitstream."""
    return 1 + NGRAM_ROW_DIM * int(bits) // 16


def _fp16_from_bits(bits16: int, device) -> torch.Tensor:
    signed = bits16 - 0x10000 if bits16 >= 0x8000 else bits16
    return torch.tensor([signed], dtype=torch.int16, device=device).view(torch.float16)


# Reimplements the mul1 codebook arithmetic of ExLlamaV3's ngram_codec
# (Copyright (c) 2025 Turboderp, MIT); see THIRD_PARTY_NOTICES.md.
def ngram_mul1_codebook(device) -> torch.Tensor:
    """The 65536-entry mul1 codebook as fp16, as exllamav3's cached table."""
    state = torch.arange(65536, device=device, dtype=torch.int64)
    prod = (state * NGRAM_MUL1) & 0xFFFFFFFF
    h = 1024.0 + (
        (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + ((prod >> 24) & 0xFF)
    ).to(torch.float32)
    k_inv = _fp16_from_bits(0x1EEE, device).to(torch.float32)
    k_bias = _fp16_from_bits(0xC931, device).to(torch.float32)
    return (h * k_inv + k_bias).to(torch.float16)


def _ngram_bit_tables(bits: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(word index, bit index) of stream bit m of element i.

    Bit m of element i lives at ring position ((i - m // K) mod ROW_DIM) * K + m % K,
    offset by one for the scale word.
    """
    i = torch.arange(NGRAM_ROW_DIM, device=device, dtype=torch.int64).unsqueeze(1)
    m = torch.arange(16, device=device, dtype=torch.int64).unsqueeze(0)
    pos = (i - m // bits) % NGRAM_ROW_DIM
    sb = pos * bits + m % bits
    return 1 + (sb >> 4), sb & 15


# Reimplements the row layout of ExLlamaV3's ngram_codec / ngram_dequant
# kernel (Copyright (c) 2025 Turboderp, MIT); see THIRD_PARTY_NOTICES.md.
def ngram_dequant_rows_torch(
    packed: torch.Tensor,
    bits: int,
    heads: torch.Tensor,
    head_bias: torch.Tensor,
    chunk: int = 8192,
) -> torch.Tensor:
    """Pure-torch twin of ``exllamav3_ext.ngram_dequant`` (fallback and test oracle).

    packed: (N, words) int16; heads: (N,) int; head_bias: (num_heads, ROW_DIM) fp16.
    Returns (N, ROW_DIM) fp16: codebook[state] * scale + head_bias[head].
    """
    device = packed.device
    widx, bidx = _ngram_bit_tables(bits, device)
    codebook = ngram_mul1_codebook(device)
    shifts = torch.arange(16, device=device, dtype=torch.int64)
    out = torch.empty(packed.shape[0], NGRAM_ROW_DIM, dtype=torch.float16, device=device)
    for s in range(0, packed.shape[0], chunk):
        p = packed[s : s + chunk]
        scale = p[:, 0].contiguous().view(torch.float16).to(torch.float32)
        words = (p.to(torch.int64) & 0xFFFF)[:, widx]
        state = (((words >> bidx) & 1) << shifts).sum(-1)
        vals = codebook[state].to(torch.float32)
        bias = head_bias[heads[s : s + chunk].to(torch.int64)].to(torch.float32)
        out[s : s + chunk] = (vals * scale.unsqueeze(1) + bias).to(torch.float16)
    return out


class Exl3EmbeddingMethod(QuantizeMethodBase):
    """Row-wise EXL3 embedding table in exllamav3's n-gram format.

    Each row is stored packed: word 0 holds the row's fp16 scale, the remaining
    ROW_DIM * K / 16 int16 words hold a tail-biting ring bitstream of 160 K-bit
    trellis states. A lookup gathers packed rows and decodes them on the fly
    (mul1 codebook * scale + per-head bias) into fp16, so the table stays at K
    bits per weight in device memory (32.6 GB for the Qwen3.8-Flash-Next table
    instead of 102 GB as bf16). Checkpoint layout under the table prefix:
    ``shard_<i>.trellis`` int16 [rows_per_shard, words], ``head_bias`` fp16
    [heads, 160], ``head_offsets`` / ``head_vocab_sizes`` int64 [heads],
    ``layer_multipliers`` int64 [n]. Shard parameters are registered as child
    modules so vLLM's AutoWeightsLoader lands them by name; they alias one
    contiguous table used for the gather.
    """

    def __init__(self, quant_config: Exl3Config, spec: dict[str, Any]) -> None:
        self.quant_config = quant_config
        self.bits = int(spec["bits"])
        self.num_shards = int(spec["num_shards"])
        self.rows_per_shard = int(spec["rows_per_shard"])
        self.num_heads = int(spec["num_heads"])
        self.words = ngram_words_per_row(self.bits)
        kernel = os.environ.get("VLLM_EXL3_NGRAM_KERNEL", "ext").strip().lower()
        if kernel not in ("ext", "torch"):
            raise ValueError(
                f"VLLM_EXL3_NGRAM_KERNEL must be 'ext' or 'torch', got {kernel!r}"
            )
        self.kernel = kernel
        self._ext = None
        self.stream_from_disk = _ngram_stream_enabled()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, extra_weight_attrs
        if int(input_size_per_partition) != NGRAM_ROW_DIM:
            raise ValueError(
                f"EXL3 n-gram rows are {NGRAM_ROW_DIM} wide; layer asks for "
                f"{int(input_size_per_partition)}"
            )
        total_rows = self.num_shards * self.rows_per_shard
        parts = [int(s) for s in output_partition_sizes]
        if int(output_size) != total_rows or parts != [total_rows]:
            raise ValueError(
                "EXL3 n-gram table geometry mismatch: vLLM built "
                f"{int(output_size)} rows (partitions {parts}) but the checkpoint "
                f"holds {self.num_shards} shards x {self.rows_per_shard} rows = {total_rows}"
            )
        _, tp_size = _resolve_tp_geometry(layer)
        if int(tp_size) != 1:
            raise RuntimeError(
                "EXL3 n-gram embeddings require VLLM_EXL3_NGRAM_STREAM=1 under "
                "tensor parallelism"
            )

        table = None
        loaded: set[int] = set()
        if not self.stream_from_disk:
            table = torch.empty(
                self.num_shards, self.rows_per_shard, self.words, dtype=torch.int16
            )
            for i in range(self.num_shards):
                shard = torch.nn.Module()
                p = Parameter(table[i], requires_grad=False)
                p.weight_loader = self._make_shard_loader(i, loaded)
                shard.register_parameter("trellis", p)
                layer.add_module(f"shard_{i}", shard)
        aux = {
            "head_bias": Parameter(
                torch.zeros(self.num_heads, NGRAM_ROW_DIM, dtype=torch.float16),
                requires_grad=False,
            ),
            "head_offsets": Parameter(
                torch.full((self.num_heads,), -1, dtype=torch.int64), requires_grad=False
            ),
            "head_vocab_sizes": Parameter(
                torch.zeros(self.num_heads, dtype=torch.int64), requires_grad=False
            ),
            "layer_multipliers": Parameter(
                torch.zeros(0, dtype=torch.int64), requires_grad=False
            ),
        }
        aux_loaded: set[str] = set()
        for name, p in aux.items():
            p.weight_loader = self._make_aux_loader(name, aux_loaded)
            layer.register_parameter(name, p)
        layer._exl3_ngram_table = table
        layer._exl3_ngram_loaded = loaded
        layer._exl3_ngram_aux_loaded = aux_loaded
        layer._exl3_ngram_dtype = params_dtype
        layer._exl3_ngram_streamed = self.stream_from_disk

    def _make_shard_loader(self, index: int, loaded: set[int]):
        rows, words, bits = self.rows_per_shard, self.words, self.bits

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
            del loaded_shard_id
            if loaded_weight.dtype != torch.int16 or tuple(loaded_weight.shape) != (rows, words):
                raise ValueError(
                    f"EXL3 n-gram shard {index}: expected int16 ({rows}, {words}) for "
                    f"K={bits}, got {loaded_weight.dtype} {tuple(loaded_weight.shape)}"
                )
            param.data.copy_(loaded_weight)
            loaded.add(index)

        return weight_loader

    def _make_aux_loader(self, name: str, aux_loaded: set[str]):
        def weight_loader(param: Parameter, loaded_weight: torch.Tensor, loaded_shard_id=None):
            del loaded_shard_id
            if name == "layer_multipliers":
                param.data = loaded_weight.to(device=param.device, dtype=param.dtype).clone()
            else:
                if tuple(loaded_weight.shape) != tuple(param.shape):
                    raise ValueError(
                        f"EXL3 n-gram {name}: expected shape {tuple(param.shape)}, "
                        f"got {tuple(loaded_weight.shape)}"
                    )
                param.data.copy_(loaded_weight.to(dtype=param.dtype))
            aux_loaded.add(name)

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        table = getattr(layer, "_exl3_ngram_table", None)
        if self.stream_from_disk:
            self._load_stream_aux(layer)
        if not self.stream_from_disk:
            loaded = layer._exl3_ngram_loaded
            missing = [i for i in range(self.num_shards) if i not in loaded]
            if missing:
                raise RuntimeError(
                    f"EXL3 n-gram table: {len(missing)} of {self.num_shards} shards never "
                    f"loaded (first missing: {missing[:8]})"
                )
        aux_missing = [
            n for n in ("head_bias", "head_offsets", "head_vocab_sizes")
            if n not in layer._exl3_ngram_aux_loaded
        ]
        if aux_missing:
            raise RuntimeError(f"EXL3 n-gram table: aux tensors never loaded: {aux_missing}")
        if (
            not self.stream_from_disk
            and layer.shard_0.trellis.data_ptr() != table.data_ptr()
        ):
            raise RuntimeError(
                "EXL3 n-gram shard parameters no longer alias the packed table; refusing to serve"
            )
        offs = layer.head_offsets.detach().cpu().tolist()
        sizes = layer.head_vocab_sizes.detach().cpu().tolist()
        total_rows = self.num_shards * self.rows_per_shard
        consistent = (
            offs[0] == 0
            and all(offs[i + 1] == offs[i] + sizes[i] for i in range(len(offs) - 1))
            and offs[-1] + sizes[-1] <= total_rows
        )
        if not consistent:
            raise RuntimeError(
                f"EXL3 n-gram head layout inconsistent with the table: offsets={offs} "
                f"sizes={sizes} rows={total_rows}"
            )
        if self.stream_from_disk:
            layer._exl3_ngram_stream_tables = self._open_stream_tables(layer)
            layer._exl3_ngram_rows = None
        else:
            layer._exl3_ngram_rows = table.view(-1, self.words)
        layer._exl3_ngram_head_offsets = layer.head_offsets.data.contiguous()
        layer._exl3_ngram_head_bias = layer.head_bias.data.contiguous()
        layer._exl3_opaque_name = _exl3_register_opaque_layer(layer, "ngram")
        if self.kernel == "ext":
            ext = load_exllamav3_ext()
            if hasattr(ext, "ngram_dequant"):
                self._ext = ext
            else:
                logger.warning(
                    "exllamav3_ext has no ngram_dequant; the EXL3 n-gram table falls back "
                    "to the torch decoder"
                )
                self.kernel = "torch"
        logger.info(
            "EXL3 n-gram embedding ready: %d shards x %d rows, K=%d, %d heads, "
            "%.2f GiB packed, kernel=%s",
            self.num_shards, self.rows_per_shard, self.bits, self.num_heads,
            (self.num_shards * self.rows_per_shard * self.words * 2) / 2**30,
            self.kernel,
        )

    def _lookup_packed(self, layer: torch.nn.Module, ids_flat: torch.Tensor) -> torch.Tensor:
        if self.stream_from_disk:
            return self._lookup_streamed(layer, ids_flat)
        return layer._exl3_ngram_rows.index_select(0, ids_flat)

    def _stream_prefixes(self, layer: torch.nn.Module) -> list[str]:
        prefix = getattr(layer, "_exl3_prefix", "")
        prefixes = [prefix]
        if "language_model.model." in prefix:
            prefixes.append(
                prefix.replace("language_model.model.", "model.language_model.", 1)
            )
        if ".model." in prefix:
            prefixes.append(prefix.replace(".model.", ".", 1))
        return prefixes

    def _load_stream_aux(self, layer: torch.nn.Module) -> None:
        """Load small n-gram metadata from turboderp's sidecar file.

        Unlike normal model weights, the EXL3 converter keeps the massive PLE
        table out of ``model.safetensors.index.json`` in
        ``ngram_embedding.safetensors``. The sidecar also owns its small lookup
        metadata, so load it before validating the normal weight-loader set.
        """
        from safetensors import safe_open
        from vllm.config import get_current_vllm_config

        root = Path(get_current_vllm_config().model_config.model)
        sidecar = root / "ngram_embedding.safetensors"
        if not sidecar.is_file():
            return
        prefixes = self._stream_prefixes(layer)
        with safe_open(sidecar, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for suffix in ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers"):
                names = [f"{prefix}.{suffix}" for prefix in prefixes]
                matches = [name for name in names if name in keys]
                if len(matches) != 1:
                    raise RuntimeError(
                        "unable to resolve exactly one EXL3 n-gram aux tensor: "
                        f"candidates={names!r}, matches={matches!r}"
                    )
                param = getattr(layer, suffix)
                loaded = handle.get_tensor(matches[0])
                if suffix == "layer_multipliers":
                    param.data = loaded.to(device=param.device, dtype=param.dtype).clone()
                else:
                    if tuple(loaded.shape) != tuple(param.shape):
                        raise ValueError(
                            f"EXL3 n-gram {suffix}: expected {tuple(param.shape)}, "
                            f"got {tuple(loaded.shape)}"
                        )
                    param.data.copy_(loaded.to(device=param.device, dtype=param.dtype))
                layer._exl3_ngram_aux_loaded.add(suffix)

    def _open_stream_tables(self, layer: torch.nn.Module) -> list[torch.Tensor]:
        """Open the converted EXL3 table through safetensors' read-only mappings."""
        from safetensors import safe_open
        from vllm.config import get_current_vllm_config

        root = Path(get_current_vllm_config().model_config.model)
        index_path = root / "model.safetensors.index.json"
        if not root.is_dir() or not index_path.is_file():
            raise FileNotFoundError(
                "VLLM_EXL3_NGRAM_STREAM requires a local model directory with "
                f"model.safetensors.index.json, got {root}"
            )
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        prefixes = self._stream_prefixes(layer)

        # turboderp EXL3 packs ship the table separately so standard vLLM
        # checkpoint iteration never attempts to materialize its 26 GiB.
        sidecar = root / "ngram_embedding.safetensors"
        if sidecar.is_file():
            handle = safe_open(sidecar, framework="pt", device="cpu")
            keys = set(handle.keys())
            tables = []
            for shard_index in range(self.num_shards):
                names = [f"{p}.shard_{shard_index}.trellis" for p in prefixes]
                matches = [name for name in names if name in keys]
                if len(matches) != 1:
                    raise RuntimeError(
                        "unable to resolve exactly one EXL3 n-gram sidecar shard: "
                        f"candidates={names!r}, matches={matches!r}"
                    )
                tensor = handle.get_tensor(matches[0])
                if tensor.dtype != torch.int16 or tuple(tensor.shape) != (
                    self.rows_per_shard,
                    self.words,
                ):
                    raise ValueError(
                        f"EXL3 n-gram sidecar shard {shard_index}: expected int16 "
                        f"{(self.rows_per_shard, self.words)}, got {tensor.dtype} "
                        f"{tuple(tensor.shape)}"
                    )
                tables.append(tensor)
            layer._exl3_ngram_stream_handles = {str(sidecar): handle}
            return tables

        handles: dict[str, Any] = {}
        tables: list[torch.Tensor] = []
        for shard_index in range(self.num_shards):
            names = [f"{p}.shard_{shard_index}.trellis" for p in prefixes]
            matches = [name for name in names if name in weight_map]
            if len(matches) != 1:
                raise RuntimeError(
                    "unable to resolve exactly one EXL3 n-gram shard: "
                    f"candidates={names!r}, matches={matches!r}"
                )
            name = matches[0]
            path = str(root / weight_map[name])
            handle = handles.get(path)
            if handle is None:
                handle = safe_open(path, framework="pt", device="cpu")
                handles[path] = handle
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.int16 or tuple(tensor.shape) != (
                self.rows_per_shard,
                self.words,
            ):
                raise ValueError(
                    f"EXL3 n-gram stream shard {shard_index}: expected int16 "
                    f"{(self.rows_per_shard, self.words)}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
            tables.append(tensor)
        # The handles own the backing mappings. Keep them strongly referenced.
        layer._exl3_ngram_stream_handles = handles
        return tables

    def _lookup_streamed(
        self, layer: torch.nn.Module, ids_flat: torch.Tensor
    ) -> torch.Tensor:
        ids_cpu = ids_flat.to(device="cpu", dtype=torch.int64).contiguous()
        if ids_cpu.numel() and (
            int(ids_cpu.min()) < 0
            or int(ids_cpu.max()) >= self.num_shards * self.rows_per_shard
        ):
            raise IndexError("EXL3 n-gram lookup index is outside the table")
        out = torch.empty((ids_cpu.numel(), self.words), dtype=torch.int16)
        shards = torch.div(ids_cpu, self.rows_per_shard, rounding_mode="floor")
        for shard_index in torch.unique(shards, sorted=True).tolist():
            mask = shards == shard_index
            local = ids_cpu[mask] - shard_index * self.rows_per_shard
            out[mask] = layer._exl3_ngram_stream_tables[shard_index].index_select(
                0, local
            )
        return out.to(device=ids_flat.device, non_blocking=True)

    def _heads_for(self, layer: torch.nn.Module, ids_flat: torch.Tensor) -> torch.Tensor:
        found = torch.searchsorted(layer._exl3_ngram_head_offsets, ids_flat, right=True) - 1
        return found.clamp_(0, self.num_heads - 1).to(torch.int32)

    def _decode(self, layer: torch.nn.Module, packed: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        bias = layer._exl3_ngram_head_bias
        if self.kernel == "ext" and self._ext is not None:
            out = torch.empty(
                packed.shape[0], NGRAM_ROW_DIM, dtype=torch.float16, device=packed.device
            )
            self._ext.ngram_dequant(packed, self.bits, heads, bias, out)
            return out
        return ngram_dequant_rows_torch(packed, self.bits, heads, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        name = getattr(layer, "_exl3_opaque_name", None)
        if name is not None and _EXL3_OPS_READY:
            return torch.ops.vllm.exl3_ngram_lookup(input_, name)
        return self._embedding_impl(layer, input_)

    def _embedding_impl(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if (
            getattr(layer, "_exl3_ngram_rows", None) is None
            and not getattr(layer, "_exl3_ngram_streamed", False)
        ):
            raise RuntimeError("EXL3 n-gram table was not finalized after weight load")
        ids = input_.reshape(-1).to(torch.int64)
        if self.stream_from_disk:
            unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            packed = self._lookup_packed(layer, unique_ids)
            heads = self._heads_for(layer, unique_ids)
            rows = self._decode(layer, packed, heads)
            out = rows.index_select(0, inverse)
            return out.to(layer._exl3_ngram_dtype).view(*input_.shape, NGRAM_ROW_DIM)
        packed = self._lookup_packed(layer, ids)
        heads = self._heads_for(layer, ids)
        out = self._decode(layer, packed, heads)
        return out.to(layer._exl3_ngram_dtype).view(*input_.shape, NGRAM_ROW_DIM)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None):
        raise NotImplementedError("EXL3 n-gram tables only support embedding lookup")


# ---------------------------------------------------------------------------
# torch.compile opacity: vLLM traces the model forward with fullgraph dynamo, which
# cannot step into exllamav3's pybind kernels. The dense linear forward and the
# n-gram lookup run behind vLLM custom ops, looked up by a stable layer name.
# ---------------------------------------------------------------------------

_EXL3_OPAQUE_LAYERS: dict[str, Any] = {}
_EXL3_OPS_READY = False


def _exl3_register_opaque_layer(layer: torch.nn.Module, kind: str) -> str:
    stable = (
        getattr(layer, "_exl3_prefix", None)
        or getattr(layer, "prefix", None)
        or getattr(layer, "layer_name", None)
        or f"id{id(layer)}"
    )
    name = f"exl3_{kind}:{stable}"
    _EXL3_OPAQUE_LAYERS[name] = layer
    return name


def _exl3_linear_forward_op(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return layer.quant_method._apply_impl(layer, x)


def _exl3_linear_forward_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    out = int(
        sum(
            getattr(
                layer, "_exl3_linear_true_out", layer._exl3_linear_output_partition_sizes
            )
        )
    )
    return x.new_empty(*x.shape[:-1], out)


def _exl3_ngram_lookup_op(ids: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return layer.quant_method._embedding_impl(layer, ids)


def _exl3_ngram_lookup_fake(ids: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _EXL3_OPAQUE_LAYERS[layer_name]
    return ids.new_empty(*ids.shape, NGRAM_ROW_DIM, dtype=layer._exl3_ngram_dtype)


def _exl3_register_custom_ops() -> bool:
    global _EXL3_OPS_READY
    if _EXL3_OPS_READY:
        return True
    try:
        try:
            from vllm.utils.torch_utils import direct_register_custom_op
        except ImportError:
            from vllm.utils import direct_register_custom_op
    except ImportError:
        return False
    try:
        if not hasattr(torch.ops.vllm, "exl3_linear_forward"):
            direct_register_custom_op(
                op_name="exl3_linear_forward",
                op_func=_exl3_linear_forward_op,
                mutates_args=[],
                fake_impl=_exl3_linear_forward_fake,
            )
        if not hasattr(torch.ops.vllm, "exl3_ngram_lookup"):
            direct_register_custom_op(
                op_name="exl3_ngram_lookup",
                op_func=_exl3_ngram_lookup_op,
                mutates_args=[],
                fake_impl=_exl3_ngram_lookup_fake,
            )
    except Exception as exc:  # pragma: no cover - registration is best effort
        logger.warning("EXL3 custom op registration failed; eager fallback: %r", exc)
        return False
    _EXL3_OPS_READY = True
    return True


if _VLLM_AVAILABLE and _TORCH_AVAILABLE:
    _exl3_register_custom_ops()



def _env_prefill_sync_rows() -> int:
    raw = os.environ.get("VLLM_EXL3_PREFILL_SYNC", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 256


_EXL3_PREFILL_SYNC = _env_prefill_sync_rows()


def _prefill_sync(rows: int) -> None:
    """Workaround for the nightly V2 runner wedge on 33..144-row prefills: serialize the
    CPU against the device before each EXL3 kernel call of a prefill step. Off unless
    VLLM_EXL3_PREFILL_SYNC=<max_rows> is set; never inside CUDA graph capture; never for
    single-row (decode) calls, so decode speed is unchanged."""
    if 1 < rows <= _EXL3_PREFILL_SYNC and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


_EXL3_GEMV_MAX_ROWS = 2
_EXL3_RECONSTRUCT_THRESHOLD = 144


def _env_int(name: str, default: int) -> int:
    """Return an int from the environment with a fallback."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except (ValueError, TypeError):
        return default


_EXL3_RECON_MIN_ROWS = _env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 17)
_EXL3_COOP_GEMM = os.environ.get("VLLM_EXL3_COOP_GEMM", "").strip() in ("1", "true", "yes")


def _dense_forward(linear, x_fp16: torch.Tensor) -> torch.Tensor:
    """Dense EXL3 forward with the wedge-prone row range kept off the cooperative GEMM.

    exllamav3 dispatches by row count: up to 2 rows run the non-cooperative GEMV
    (exl3_gemv_int8; the QTIP gemv needs K in 2..4 so it never applies to K=5),
    3..144 rows run the cooperative trellis GEMM (cudaLaunchCooperativeKernel,
    grid barriers over a shared lock buffer), and above 144 rows LinearEXL3
    reconstructs the weight and runs hgemm. On the vLLM nightly V2 model runner
    the cooperative GEMM wedged the engine on every 33..144-token prefill and,
    through those prefills, every long MTP run. Routing rows >= 17 through the
    reconstruct path removed the wedge: a 4-worker MTP k=2 stress that wedged
    the baseline in 161 s ran clean for 45 minutes (1085 requests) with decode
    speed unchanged. Rows 3..16 (single-request MTP steps and small batches)
    keep exllamav3's dispatch; that range ran clean in the same 45 minutes and
    the reconstruct path would cost about 10x per call there.

    VLLM_EXL3_RECONSTRUCT_MIN_ROWS moves the threshold (17 by default);
    VLLM_EXL3_COOP_GEMM=1 restores the old dispatch for A/B runs.
    """
    rows = int(x_fp16.shape[0])
    if (not _EXL3_COOP_GEMM) and _EXL3_RECON_MIN_ROWS <= rows <= _EXL3_RECONSTRUCT_THRESHOLD:
        return linear.forward(x_fp16, {"reconstruct": True}, out_dtype=torch.float32)
    return linear.forward(x_fp16, {}, out_dtype=torch.float32)


class Exl3LinearMethod(LinearMethodBase):
    """Non-routed (dense) EXL3 linear method for QKV/MLP dense projections.

    This method handles trellis/suh/svh/mcg parameters for non-routed dense
    linear layers, building LinearEXL3 objects after weight loading and applying
    them with proper TP slicing and shard concatenation.
    """

    def __init__(self, quant_config: Exl3Config, bits: int | None = None) -> None:
        self.quant_config = quant_config
        self.bits = int(bits) if bits is not None else quant_config.bits
        self._logged = False

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear,
            RowParallelLinear,
            QKVParallelLinear,
            MergedColumnParallelLinear,
        )
        from vllm.distributed import get_tensor_model_parallel_world_size

        # Determine layer type and shard behavior
        n_shards = len(output_partition_sizes)
        is_row_parallel = isinstance(layer, RowParallelLinear)
        is_col_parallel = isinstance(layer, ColumnParallelLinear)
        is_qkv_parallel = isinstance(layer, QKVParallelLinear)
        is_merged_col_parallel = isinstance(layer, MergedColumnParallelLinear)

        # For column-parallel: output dimension is sharded (each shard has different outputs)
        # For row-parallel: input dimension is sharded (each shard has same input, different outputs)
        if is_row_parallel:
            # Input is partitioned across ranks, each rank gets full input height
            in_per_partition = input_size_per_partition
        else:
            # Column-parallel or unsharded: each rank gets full input
            in_per_partition = input_size_per_partition

        # Get bf16_shards from config (may be empty)
        bf16_shards = self.quant_config._bf16_shards_for(getattr(layer, "prefix", ""))
        if bf16_shards:
            _, tp_size = _resolve_tp_geometry(layer)
            if tp_size > 1:
                raise RuntimeError(
                    f"EXL3 bf16 shards are not supported with TP size > 1; tp_size={tp_size}"
                )

        # K words per shard
        k_words = self.bits * 16

        # EXL3 pads both matrix dims to multiples of 128 (zeros at the end;
        # padded output columns carry svh = 0, padded input rows only see
        # zero-extended inputs). Allocate the padded geometry, load the
        # checkpoint tensors whole, and pad / trim activations in apply().
        true_in = int(in_per_partition)
        true_out_sizes = [int(s) for s in output_partition_sizes]
        in_per_partition = _exl3_pad128(true_in)
        output_partition_sizes = [_exl3_pad128(s) for s in true_out_sizes]
        padded = in_per_partition != true_in or output_partition_sizes != true_out_sizes
        if padded and bf16_shards:
            raise NotImplementedError(
                "EXL3 padded linear geometry is not supported with bf16 shards: "
                f"in={true_in} out={true_out_sizes}"
            )
        if padded:
            # vLLM registers the bias after create_weights with this instance's
            # weight_loader; a padded checkpoint bias is trimmed to the true size.
            _orig_loader = layer.weight_loader

            def _bias_trim_loader(param, loaded_weight, *args, **kwargs):
                if param.dim() == 1 and int(loaded_weight.shape[0]) > int(param.shape[0]):
                    loaded_weight = loaded_weight[: int(param.shape[0])]
                return _orig_loader(param, loaded_weight, *args, **kwargs)

            layer.weight_loader = _bias_trim_loader

        # Validate tile alignment for all shards
        for i, out_size in enumerate(output_partition_sizes):
            if in_per_partition % 16 or out_size % 16:
                raise ValueError(
                    f"EXL3 trellis tiles are 16-wide; "
                    f"shard {i}: in={in_per_partition} out={out_size}"
                )

        in_tiles = in_per_partition // 16
        out_tiles_list = [s // 16 for s in output_partition_sizes]
        total_out_tiles = sum(out_tiles_list)

        # Allocate fused trellis covering all shards (dim1 will be narrow per-shard)
        trellis_param = Parameter(
            torch.zeros(in_tiles, total_out_tiles, k_words, dtype=torch.int16),
            requires_grad=False,
        )
        # Per-shard suh (one per shard, each covers this rank's input partition)
        suh_param = Parameter(
            torch.zeros(n_shards, in_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        # Per-shard svh (one per shard, concatenated)
        svh_param = Parameter(
            torch.zeros(sum(output_partition_sizes), dtype=torch.float16),
            requires_grad=False,
        )
        # Per-shard mcg and mul1 markers (both registered, one will be nonzero)
        mcg_param = Parameter(
            torch.zeros(n_shards, 1, dtype=torch.int32),
            requires_grad=False,
        )
        mul1_param = Parameter(
            torch.zeros(n_shards, 1, dtype=torch.int32),
            requires_grad=False,
        )

        # Staging parameter for bf16 shards: rows are concatenated bf16 weights
        bf16_rows = sum(output_partition_sizes[i] for i in bf16_shards)
        weight_param = Parameter(
            torch.empty(bf16_rows, in_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("trellis", trellis_param)
        layer.register_parameter("suh", suh_param)
        layer.register_parameter("svh", svh_param)
        layer.register_parameter("mcg", mcg_param)
        layer.register_parameter("mul1", mul1_param)
        layer.register_parameter("weight", weight_param)

        # Custom weight loader
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        set_weight_attrs(trellis_param, extra)
        set_weight_attrs(suh_param, extra)
        set_weight_attrs(svh_param, extra)
        set_weight_attrs(mcg_param, extra)
        set_weight_attrs(mul1_param, extra)
        set_weight_attrs(weight_param, extra)

        # vLLM calls ``weight_loader(param, loaded_weight[, shard_id])`` and
        # never passes the checkpoint name, so bind the tensor kind per param.
        for suffix, p in (
            ("trellis", trellis_param),
            ("suh", suh_param),
            ("svh", svh_param),
            ("mcg", mcg_param),
            ("mul1", mul1_param),
        ):
            p.weight_loader = self._make_weight_loader(
                suffix,
                n_shards,
                output_partition_sizes,
                is_row_parallel,
                bf16_shards,
                layer,
                is_qkv_parallel,
            )
        weight_param.weight_loader = self._make_weight_loader(
            "weight",
            n_shards,
            output_partition_sizes,
            is_row_parallel,
            bf16_shards,
            layer,
            is_qkv_parallel,
        )

        # Store metadata
        layer._exl3_linear_n_shards = n_shards
        layer._exl3_linear_output_partition_sizes = output_partition_sizes
        layer._exl3_linear_input_size_per_partition = in_per_partition
        layer._exl3_linear_is_row_parallel = is_row_parallel
        layer._exl3_linear_is_qkv = is_qkv_parallel
        layer._exl3_linear_is_merged = is_merged_col_parallel
        layer._exl3_linear_bf16_shards = bf16_shards
        layer._exl3_linear_padded = padded
        layer._exl3_linear_true_in = true_in
        layer._exl3_linear_true_out = true_out_sizes

    def _make_weight_loader(
        self,
        suffix,
        n_shards,
        output_partition_sizes,
        is_row_parallel,
        bf16_shards,
        layer=None,
        is_qkv_parallel=False,
    ):
        """Create a weight_loader closure for EXL3 linear parameters."""

        def weight_loader(
            param: Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | int | None = None,
        ) -> None:
            tp_rank, tp_size = _resolve_tp_geometry(layer, param)

            # One checkpoint tensor may span several consecutive shards; vLLM's
            # WeightsMapper says so with a tuple of shard ids (Qwen3.5/4
            # in_proj_qkv -> in_proj_qkvz shards (0, 1, 2)). Split it along the
            # output dimension at the shard boundaries and load each piece.
            span_ids = None
            if isinstance(loaded_shard_id, (tuple, list)):
                span_ids = [int(i) for i in loaded_shard_id]
            elif loaded_shard_id is None and n_shards > 1:
                # Already-fused checkpoint tensor on a merged linear: an
                # output-sized tensor covering every shard is split; per-input
                # tensors and markers apply to every shard.
                if suffix in ("suh", "mcg", "mul1"):
                    span_ids = list(range(n_shards))
                else:
                    loaded_out = (
                        int(loaded_weight.shape[1]) * 16
                        if suffix == "trellis"
                        else int(loaded_weight.shape[0])
                    )
                    if loaded_out == sum(output_partition_sizes) and loaded_out != int(
                        output_partition_sizes[0]
                    ):
                        span_ids = list(range(n_shards))
            if span_ids is not None:
                ids = span_ids
                if (
                    not ids
                    or ids != list(range(ids[0], ids[0] + len(ids)))
                    or ids[-1] >= n_shards
                ):
                    raise ValueError(
                        f"EXL3 linear: unsupported shard id span {loaded_shard_id} "
                        f"for n_shards={n_shards}"
                    )
                if suffix in ("suh", "mcg", "mul1"):
                    for i in ids:
                        weight_loader(param, loaded_weight, i)
                    return
                loaded_out = (
                    int(loaded_weight.shape[1]) * 16
                    if suffix == "trellis"
                    else int(loaded_weight.shape[0])
                )
                span = sum(output_partition_sizes[i] for i in ids)
                if loaded_out != span:
                    # A fused QKV checkpoint tensor is global, whereas the
                    # parameter being loaded is rank-local. Split at the
                    # global Q/K/V boundaries first, then let the ordinary
                    # per-shard path apply its existing TP slicing. This must
                    # happen before the rank-local span validation below.
                    _, tp_size = _resolve_tp_geometry(layer, param)
                    true_out_sizes = getattr(
                        layer, "_exl3_linear_true_out", output_partition_sizes
                    )
                    global_span = sum(int(true_out_sizes[i]) * int(tp_size) for i in ids)
                    if loaded_out != global_span:
                        raise RuntimeError(
                            f"EXL3 linear load: {suffix} tensor covers {loaded_out} outputs "
                            f"but shards {ids} total {span} (global total {global_span})"
                        )
                    start = 0
                    for i in ids:
                        size = int(true_out_sizes[i]) * int(tp_size)
                        if suffix == "trellis":
                            piece = loaded_weight[:, start // 16 : (start + size) // 16, :]
                        else:
                            piece = loaded_weight[start : start + size]
                        weight_loader(param, piece, i)
                        start += size
                    return
                start = 0
                for i in ids:
                    size = output_partition_sizes[i]
                    if suffix == "trellis":
                        piece = loaded_weight[:, start // 16 : (start + size) // 16, :]
                    else:
                        piece = loaded_weight[start : start + size]
                    weight_loader(param, piece, i)
                    start += size
                return

            # Map shard_id to shard index
            shard_idx = 0
            if loaded_shard_id is not None:
                if isinstance(loaded_shard_id, str):
                    # "q", "k", "v" for QKV layers
                    shard_map = {"q": 0, "k": 1, "v": 2}
                    if loaded_shard_id not in shard_map:
                        raise ValueError(
                            f"unknown shard_id={loaded_shard_id} for EXL3 linear"
                        )
                    shard_idx = shard_map[loaded_shard_id]
                elif isinstance(loaded_shard_id, int):
                    shard_idx = loaded_shard_id
            if shard_idx >= n_shards:
                raise ValueError(
                    f"shard_idx={shard_idx} out of range for n_shards={n_shards}"
                )

            # Special handling for weight (bf16 staging) and markers
            if suffix in ("weight", "mcg", "mul1"):
                # Weight parameter: only load bf16 shards, discard EXL3 shards
                if suffix == "weight":
                    # Check shape matches the expected shard size
                    expected_out = output_partition_sizes[shard_idx]
                    expected_in = param.shape[1]
                    loaded = loaded_weight.detach().contiguous()
                    loaded_shape = loaded.shape
                    if is_qkv_parallel and not is_row_parallel:
                        total_out = int(loaded_shape[0])
                        shard_tp_size = max(1, total_out // expected_out)
                        shard_tp_rank = tp_rank // max(1, tp_size // shard_tp_size)
                    else:
                        shard_tp_size = tp_size
                        shard_tp_rank = tp_rank
                    if tuple(loaded_shape) == (expected_out, expected_in):
                        tp_sharded = loaded
                    else:
                        # Row-parallel input is sharded on dim 1; column-parallel
                        # output is sharded on dim 0.
                        slice_dim = 1 if is_row_parallel else 0
                        tp_sharded = _narrow_tp(
                            loaded,
                            slice_dim,
                            shard_tp_rank,
                            shard_tp_size,
                        )
                    if tuple(tp_sharded.shape) != (expected_out, expected_in):
                        raise RuntimeError(
                            f"EXL3 weight load shape mismatch shard={shard_idx}: "
                            f"expected ({expected_out},{expected_in}) but got {tuple(loaded.shape)} "
                            f"(after TP: {tuple(tp_sharded.shape)})"
                        )
                    # If this shard is in bf16_shards, copy; otherwise discard
                    if shard_idx in bf16_shards:
                        bf16_idx = bf16_shards.index(shard_idx)
                        bf16_row_start = sum(output_partition_sizes[i] for i in bf16_shards[:bf16_idx])
                        bf16_row_end = bf16_row_start + expected_out
                        param.data[bf16_row_start:bf16_row_end].copy_(tp_sharded)
                    # else: discard this EXL3 shard's stale BF16 weight
                    return
                else:
                    # Marker (mcg or mul1): store the value (will be 0 if marker not present)
                    dest = param.data[shard_idx]
                    if tuple(dest.shape) != (1,):
                        raise RuntimeError(
                            f"EXL3 {suffix} marker shape mismatch: expected (1,) got {tuple(dest.shape)}"
                        )
                    loaded_val = loaded_weight.detach().item() if loaded_weight.numel() > 0 else 0
                    dest[0] = int(loaded_val)
                    return

            # Normal EXL3 suffix handling (trellis, suh, svh)
            loaded = loaded_weight.detach().contiguous()

            expected_out = output_partition_sizes[shard_idx]
            if is_qkv_parallel and not is_row_parallel:
                total_out = (
                    int(loaded.shape[1]) * 16
                    if suffix == "trellis"
                    else int(loaded.shape[0])
                )
                shard_tp_size = max(1, total_out // expected_out)
                shard_tp_rank = tp_rank // max(1, tp_size // shard_tp_size)
            else:
                shard_tp_size = tp_size
                shard_tp_rank = tp_rank

            # Some GQA K/V checkpoint tensors are already rank-local because
            # their heads are replicated rather than column-sharded. Recognize
            # that geometry before applying normal TP slicing; otherwise a
            # local 320-column tensor would be narrowed to 160 a second time.
            true_in = getattr(layer, "_exl3_linear_true_in", None)
            true_out_sizes = getattr(layer, "_exl3_linear_true_out", None)
            already_local = False
            if true_in is not None and true_out_sizes is not None:
                true_in = int(true_in)
                true_out = int(true_out_sizes[shard_idx])
                if suffix == "trellis":
                    already_local = tuple(loaded.shape[:2]) == (
                        true_in // 16,
                        true_out // 16,
                    )
                elif suffix == "suh":
                    already_local = int(loaded.shape[0]) == true_in
                else:  # svh
                    already_local = int(loaded.shape[0]) == true_out

            if already_local:
                sharded = loaded
            elif is_row_parallel:
                # Row-parallel: input is sharded, trellis dim 0 and suh dim 0.
                sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            else:
                # Column-parallel: output is sharded, trellis dim 1 and svh dim 0.
                sharded = shard_exl3_col(
                    loaded, suffix, shard_tp_rank, shard_tp_size
                )

            # Copy into the right location
            if suffix == "trellis":
                # Trellis is fused; narrow dim1 for this shard
                out_tiles_start = sum(s // 16 for s in output_partition_sizes[:shard_idx])
                out_tiles_end = out_tiles_start + output_partition_sizes[shard_idx] // 16
                dest = param.data[:, out_tiles_start:out_tiles_end, :]
            elif suffix == "suh":
                # Suh per-shard
                dest = param.data[shard_idx]
            elif suffix == "svh":
                # Svh is concatenated; slice for this shard
                out_start = sum(output_partition_sizes[:shard_idx])
                out_end = out_start + output_partition_sizes[shard_idx]
                dest = param.data[out_start:out_end]
            else:
                raise ValueError(f"unknown EXL3 suffix={suffix}")

            if tuple(dest.shape) != tuple(sharded.shape):
                # Qwen Flash shared experts split a 320-wide gate/up projection
                # into two merged-column shards. EXL3 treats each shard as an
                # independently 128-padded matrix, so retain the checkpoint data
                # at the origin and leave the added trellis/suh/svh entries zero.
                # A zero svh makes padded output columns identically zero before
                # _apply_impl trims them. The same rule also covers padded input
                # rows, which receive zero-extended activations.
                if not (
                    getattr(layer, "_exl3_linear_padded", False)
                    and dest.dim() == sharded.dim()
                    and all(src <= target for src, target in zip(sharded.shape, dest.shape))
                ):
                    raise RuntimeError(
                        f"EXL3 linear load shape mismatch shard={shard_idx} "
                        f"suffix={suffix}: dest {tuple(dest.shape)} != "
                        f"loaded {tuple(sharded.shape)}"
                    )
                dest[tuple(slice(0, size) for size in sharded.shape)].copy_(sharded)
            else:
                dest.copy_(sharded)

        return weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "trellis"):
            return

        # Get bf16 shards and verify exactly one marker per EXL3 shard
        n_shards = int(layer._exl3_linear_n_shards)
        output_sizes = layer._exl3_linear_output_partition_sizes
        bf16_shards = getattr(layer, "_exl3_linear_bf16_shards", [])

        mcg_vals = layer.mcg.reshape(-1)
        mul1_vals = layer.mul1.reshape(-1)

        for i in range(n_shards):
            # Skip marker checks for bf16 shards - they don't use LinearEXL3
            if i in bf16_shards:
                continue
            mcg_is_set = mcg_vals[i].item() != 0
            mul1_is_set = mul1_vals[i].item() != 0
            if mcg_is_set and mul1_is_set:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: both mcg and mul1 markers are set; "
                    f"exactly one codebook marker must be present"
                )
            if not mcg_is_set and not mul1_is_set:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: neither mcg nor mul1 marker is set; "
                    f"exactly one codebook marker must be present"
                )
            # Verify marker value
            if mcg_is_set and mcg_vals[i].item() != MCG_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: mcg marker is {mcg_vals[i].item()}, "
                    f"expected {MCG_MARKER_SIGNED_INT32}"
                )
            if mul1_is_set and mul1_vals[i].item() != MUL1_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: mul1 marker is {mul1_vals[i].item()}, "
                    f"expected {MUL1_MARKER_SIGNED_INT32}"
                )

        # Build LinearEXL3 objects for EXL3 shards only (skip bf16 shards)
        linears = []
        for i in range(n_shards):
            if i in bf16_shards:
                # bf16 shards don't use LinearEXL3; store None as placeholder
                linears.append(None)
                continue
            out_tiles_start = sum(s // 16 for s in output_sizes[:i])
            out_tiles_end = out_tiles_start + output_sizes[i] // 16
            trellis_shard = layer.trellis[:, out_tiles_start:out_tiles_end, :].contiguous()
            suh_shard = layer.suh[i].contiguous()
            svh_shard = layer.svh[
                sum(output_sizes[:i]) : sum(output_sizes[: i + 1])
            ].contiguous()
            mcg_shard = layer.mcg[i].contiguous() if mcg_vals[i].item() != 0 else None
            mul1_shard = layer.mul1[i].contiguous() if mul1_vals[i].item() != 0 else None

            linear = make_linear_exl3(
                trellis_shard, suh_shard, svh_shard, mcg_shard, mul1_shard, out_dtype=torch.float16
            )
            linears.append(linear)

        layer._exl3_linears = linears
        layer._exl3_opaque_name = _exl3_register_opaque_layer(layer, "linear")

        # Keep bf16 weights if present, remove weight staging param if all loaded
        if bf16_shards and hasattr(layer, "weight"):
            bf16_rows = sum(output_sizes[i] for i in bf16_shards)
            if bf16_rows > 0:
                layer._exl3_bf16_weight = layer.weight.data.clone()
            # Delete weight staging param only if it has rows; empty param stays as placeholder
            if layer.weight.data.shape[0] > 0:
                try:
                    delattr(layer, "weight")
                except Exception:
                    pass
        elif hasattr(layer, "weight"):
            # No bf16 shards, delete the staging param
            try:
                delattr(layer, "weight")
            except Exception:
                pass

        # Free fused parameters to avoid memory doubling
        for param_name in ("trellis", "suh", "svh", "mcg", "mul1"):
            if hasattr(layer, param_name):
                try:
                    delattr(layer, param_name)
                except Exception:
                    pass

    def apply(
        self,
        layer,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        name = getattr(layer, "_exl3_opaque_name", None)
        if name is not None and _EXL3_OPS_READY:
            y = torch.ops.vllm.exl3_linear_forward(x, name)
        else:
            y = self._apply_impl(layer, x)
        if bias is not None:
            y = y + bias
        return y

    def _apply_impl(self, layer, x: torch.Tensor) -> torch.Tensor:
        linears = getattr(layer, "_exl3_linears", None)
        if _EXL3_PREFILL_SYNC:
            _prefill_sync(int(x.numel() // x.shape[-1]))
        if not linears:
            raise RuntimeError("EXL3 linear layers were not built after weight load")

        # x shape: (batch, in_features) or (batch, ..., in_features)
        # Flatten to 2D: (rows, in_features)
        orig_shape = x.shape
        if len(orig_shape) > 2:
            # Multi-dim input: flatten to (rows, in)
            rows = 1
            for d in orig_shape[:-1]:
                rows *= d
            x_2d = x.reshape(rows, orig_shape[-1])
        else:
            x_2d = x

        # Cast to contiguous fp16 for EXL3 shards
        x_fp16 = x_2d.to(torch.float16).contiguous()
        if getattr(layer, "_exl3_linear_padded", False):
            pad_in = int(layer._exl3_linear_input_size_per_partition) - int(x_fp16.shape[1])
            if pad_in > 0:
                x_fp16 = F.pad(x_fp16, (0, pad_in))

        # Get bf16 shards and weight if present
        bf16_shards = getattr(layer, "_exl3_linear_bf16_shards", [])
        bf16_weight = getattr(layer, "_exl3_bf16_weight", None)
        output_sizes = layer._exl3_linear_output_partition_sizes
        n_shards = len(linears)

        # Run each shard in declared order
        outputs = []
        for i in range(n_shards):
            if i in bf16_shards:
                # BF16 shard: use dense linear
                if bf16_weight is None:
                    raise RuntimeError(
                        f"EXL3 bf16 shard {i} but _exl3_bf16_weight is missing"
                    )
                bf16_idx = bf16_shards.index(i)
                out_start = sum(output_sizes[j] for j in bf16_shards[:bf16_idx])
                out_end = out_start + output_sizes[i]
                w_shard = bf16_weight[out_start:out_end]
                out = F.linear(x_2d, w_shard).to(dtype=torch.float32)
                outputs.append(out)
            else:
                # EXL3 shard
                linear = linears[i]
                if linear is None:
                    raise RuntimeError(f"EXL3 linear shard {i} is None")
                out = _dense_forward(linear, x_fp16)
                outputs.append(out)

        # Concatenate shards along output dimension
        if len(outputs) > 1:
            y = torch.cat(outputs, dim=1)
        else:
            y = outputs[0]

        # Trim padded output columns (svh = 0 there, so they are zeros)
        if getattr(layer, "_exl3_linear_padded", False):
            y = y[:, : sum(layer._exl3_linear_true_out)]

        # Cast back to input dtype
        y = y.to(dtype=x.dtype)

        # Restore original shape
        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], y.shape[-1])

        return y
