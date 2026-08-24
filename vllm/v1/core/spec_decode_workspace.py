# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[FORK] Analytical sizing for the speculative-decode *verify* working set that
KV-cache memory profiling under-counts at large ``num_speculative_tokens``.

Kept torch-free on purpose: the pure formula (:func:`spec_verify_reserve_bytes`)
is unit-tested against the measured OOM facts without importing CUDA/torch. The
config-facing wrapper lives in ``kv_cache_utils`` and only pulls scalars out of
``VllmConfig`` before delegating here.

Root cause (see ``docs/exp039-scoped-drafter-design.md`` §5, K-sizing finding #2)
-----------------------------------------------------------------------------
``GPUModelRunner.profile_run`` runs a *prefill*-shaped ``_dummy_run`` and then
``_dummy_sampler_run``. In the spec-decode branch of ``_dummy_sampler_run`` the
rejection sampler is exercised with ``draft_token_ids = [[0]] * num_reqs`` — i.e.
**one** draft token per request (``logits = randn(2 * num_reqs, vocab)``), and
``compute_logits`` in the prefill dummy run only produces ``num_reqs`` logit rows
(one per request). A *real* decode step with ``num_speculative_tokens = K`` must
instead verify ``K`` draft positions + 1 bonus position per request, so it:

* runs ``compute_logits`` over ``logits_indices`` of length ``(1 + K) * num_reqs``
  (``gpu_model_runner._calc_spec_decode_metadata`` builds target+bonus indices),
  producing a ``((1 + K) * num_reqs, vocab)`` fp32 tensor, and
* drives the rejection sampler (``vllm/v1/sample/rejection_sampler.py``) whose
  working set is several concurrent full-vocab **fp32** tensors sized by the
  ``K * num_reqs`` target positions: ``raw_target_logits`` (fancy-index copy),
  its ``.clone()``, the top-k/top-p sort scratch, and the ``target_probs``
  softmax.

None of that full-width verify peak is materialised during profiling, so it
lands lazily on the *first real decode step* — after the KV cache has already
been sized to consume the rest of VRAM — and OOMs post-profiling. The gap scales
with ``(K - 1) * num_reqs`` (the per-request verify width beyond the profiled
K=1 baseline), which is exactly why it is invisible at K=2 but ~5 GiB at K=16,
and why it shrinks ~4x when ``max_num_seqs`` drops 16 -> 4.

Fix strategy: subtract an analytical reserve for this working set from the KV
budget *before* ``num_gpu_blocks`` is derived, mirroring
``_turboquant_prefill_workspace_reserve_bytes``. Unlike the turboquant
continuation workspace (a ``WorkspaceManager`` arena that can be allocated at
load time and therefore needs an add-back in ``determine_available_memory`` to
avoid double-counting), the verify buffers are plain transient torch allocations
that profiling **never** triggers at full width — so there is nothing to add
back and the reserve is applied exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

# Profiling (`_dummy_sampler_run`) exercises the rejection sampler with one draft
# token per request, i.e. `num_draft (=1) + bonus (=1) = 2` logit positions per
# request. This is the width already reflected in the profiled torch peak.
PROFILED_BASELINE_WIDTH = 2

# fp32: `compute_logits` returns float32 and the rejection sampler upcasts
# (`raw_target_logits.to(torch.float32)`, `softmax(dtype=torch.float32)`,
# `torch.zeros_like(logits, dtype=torch.float32)`).
VERIFY_DTYPE_BYTES = 4

# Mechanistic lower bound on the number of concurrent full-vocab fp32 tensors the
# verify path holds *beyond* the profiled K=1 baseline, counted from the code:
#   1x compute_logits verify-output delta
#   1x raw_target_logits (fancy-index copy)
#   1x target_logits clone
#   2x top-k/top-p sort scratch (sorted values + scatter/cumsum)
#   1x target_probs softmax
# ~= 6.  This is the floor; see OVERSHOOT_MULT_DEFAULT for why the default is
# larger.
MECHANISTIC_OVERSHOOT_MULT = 6

# Default multiplier. The mechanistic verify buffers (~6x) explain ~1.4 GiB of
# the measured ~5 GiB post-profiling overshoot at K=16 / max_num_seqs=16 /
# vocab=248320. The remainder is *not* verify logits (the arithmetic ceiling of
# the verify buffers is ~2 GiB) — it is the un-instrumented residual that also
# scales with the (1+K)-wide decode batch: the GDN/Mamba align-mode decode scan
# workspace and PyTorch caching-allocator fragmentation. Until that residual is
# attributed with an instrumented boot, the default reserves the full observed
# envelope so a high-util / high-seqs boot does not OOM. Operators who have
# separately accounted for the residual can lower this toward
# MECHANISTIC_OVERSHOOT_MULT.
OVERSHOOT_MULT_DEFAULT = 24


@dataclass(frozen=True)
class SpecVerifyReserve:
    """Breakdown of the spec-decode verify reserve (for logging / tests)."""

    num_speculative_tokens: int
    max_num_seqs: int
    vocab_size: int
    overshoot_mult: int
    dtype_bytes: int
    profiled_baseline_width: int

    @property
    def per_req_delta_width(self) -> int:
        """Per-request verify logit positions the profiler misses.

        Real width is ``K + 1`` (K target + 1 bonus); profiling covers
        ``profiled_baseline_width`` (=2).
        """
        return max(0, (self.num_speculative_tokens + 1) - self.profiled_baseline_width)

    @property
    def delta_positions(self) -> int:
        return self.per_req_delta_width * max(0, self.max_num_seqs)

    @property
    def base_bytes(self) -> int:
        """One full-vocab fp32 row-group over the missed positions."""
        return self.delta_positions * max(0, self.vocab_size) * self.dtype_bytes

    @property
    def reserve_bytes(self) -> int:
        return self.overshoot_mult * self.base_bytes


def spec_verify_reserve(
    num_speculative_tokens: int,
    max_num_seqs: int,
    vocab_size: int,
    *,
    overshoot_mult: int = OVERSHOOT_MULT_DEFAULT,
    dtype_bytes: int = VERIFY_DTYPE_BYTES,
    profiled_baseline_width: int = PROFILED_BASELINE_WIDTH,
) -> SpecVerifyReserve:
    """Build the reserve breakdown (does not gate on any env)."""
    return SpecVerifyReserve(
        num_speculative_tokens=int(num_speculative_tokens),
        max_num_seqs=int(max_num_seqs),
        vocab_size=int(vocab_size),
        overshoot_mult=int(overshoot_mult),
        dtype_bytes=int(dtype_bytes),
        profiled_baseline_width=int(profiled_baseline_width),
    )


def spec_verify_reserve_bytes(
    num_speculative_tokens: int,
    max_num_seqs: int,
    vocab_size: int,
    *,
    overshoot_mult: int = OVERSHOOT_MULT_DEFAULT,
    dtype_bytes: int = VERIFY_DTYPE_BYTES,
    profiled_baseline_width: int = PROFILED_BASELINE_WIDTH,
) -> int:
    """Per-rank VRAM (bytes) to reserve for the spec-decode verify working set.

    Returns 0 for the non-spec / K<=1 case (nothing beyond the profiled
    baseline) and for any degenerate input.

    The reserve models the peak of the concurrent full-vocab fp32 verify buffers
    that profiling under-counts, scaled by ``(K - 1) * max_num_seqs`` — the
    per-request verify width beyond the profiled K=1 baseline, times the batch.
    """
    if (
        num_speculative_tokens is None
        or max_num_seqs is None
        or vocab_size is None
        or overshoot_mult is None
        or dtype_bytes is None
        or profiled_baseline_width is None
    ):
        # Any None kwarg is a degenerate input; return 0 before int() so the
        # documented "0 for degenerate input" contract holds instead of TypeError.
        return 0
    if (
        int(num_speculative_tokens) <= 1
        or int(max_num_seqs) <= 0
        or int(vocab_size) <= 0
        or int(overshoot_mult) <= 0
        or int(dtype_bytes) <= 0
    ):
        return 0
    return spec_verify_reserve(
        num_speculative_tokens,
        max_num_seqs,
        vocab_size,
        overshoot_mult=overshoot_mult,
        dtype_bytes=dtype_bytes,
        profiled_baseline_width=profiled_baseline_width,
    ).reserve_bytes


def spec_verify_reserve_decision(
    *,
    speculative_present: bool,
    enabled: bool,
    num_speculative_tokens: int,
    max_num_seqs: int,
    vocab_size: int,
    overshoot_mult: int,
) -> tuple[int, str]:
    """Return ``(reserve_bytes, human_reason)`` for the spec-verify reserve.

    Pure (scalars in, tuple out) so the *always-on* boot diagnostic and the
    actual subtraction agree by construction and can be unit-tested without an
    engine. ``reason`` is safe to log verbatim; it says either ``applied ...`` or
    ``skipped: <why>`` so a boot is attributable even when a *different* phase
    (e.g. cudagraph-memory profiling) OOMs before the reserve is used.
    """
    if not speculative_present:
        return 0, "skipped: no speculative_config"
    if not enabled:
        return 0, "skipped: VLLM_SPEC_RESERVE_VERIFY_WORKSPACE=0"
    if int(num_speculative_tokens) <= 1:
        return 0, (
            f"skipped: num_speculative_tokens={num_speculative_tokens} <= 1 "
            "(nothing beyond the profiled K=1 baseline)"
        )
    if int(overshoot_mult) <= 0:
        return 0, (
            f"skipped: VLLM_SPEC_VERIFY_OVERSHOOT_MULT={overshoot_mult} <= 0"
        )
    if int(max_num_seqs) <= 0 or int(vocab_size) <= 0:
        return 0, (
            f"skipped: degenerate (max_num_seqs={max_num_seqs}, "
            f"vocab_size={vocab_size})"
        )
    b = spec_verify_reserve_bytes(
        num_speculative_tokens,
        max_num_seqs,
        vocab_size,
        overshoot_mult=overshoot_mult,
    )
    reason = (
        f"applied: K={num_speculative_tokens} max_num_seqs={max_num_seqs} "
        f"vocab={vocab_size} mult={overshoot_mult} "
        f"({b / (1 << 30):.3f} GiB)"
    )
    return b, reason


def spec_verify_last_stage_mask(
    pp_size: int, inner_size: int, n_workers: int
) -> list[bool]:
    """Per-worker mask of last-pipeline-stage membership.

    Worker lists follow the parallel_state rank layout (DP x) PP x PCP x TP
    with TP innermost. A per-DP-rank executor covers PP*PCP*TP workers;
    ``external_launcher`` folds DP in outermost, so the last stage is NOT a
    contiguous tail slice there -- each DP group has its own last stage.
    Deriving the stage from the PCP*TP inner block size is correct in both
    modes: ``stage(idx) = (idx // (pcp*tp)) % pp``.

    Defensive: if n_workers is not a multiple of inner_size * pp_size the
    layout assumption is off, so reserve on every worker (safe over-reserve
    rather than a missed reserve on the rank that actually needs it).
    """
    if pp_size <= 1:
        return [True] * n_workers
    if n_workers % (inner_size * pp_size) != 0:
        return [True] * n_workers
    return [(idx // inner_size) % pp_size == pp_size - 1 for idx in range(n_workers)]
