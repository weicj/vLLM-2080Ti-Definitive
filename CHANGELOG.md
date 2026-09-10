# Changelog

This changelog tracks the fork release version for vLLM 2080 Ti Definitive
Edition. It is separate from the upstream vLLM package version.

## Unreleased

- Keeps `fast`/`aggressive` launcher modes on PIECEWISE whenever native MTP /
  speculative decoding is enabled. Full decode CUDA-graph replay for hybrid
  Mamba/GDN layers is only exposed through the documented unsafe peak-throughput
  route, because the graph-captured recurrent-state update topology is reused
  across changing speculative acceptance patterns and deterministically scrambles
  the model context (e.g. `123 + 456` answered as `1 + 2 = 3`). Explicit
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1` still opts into the old route.
- Shares the target FlashInfer workspace buffer with the draft attention builders.
  The draft layer previously allocated a second lazily-sized
  `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` (~394 MiB) workspace after the KV cache
  had claimed the remaining GPU memory, which could OOM the first request at high
  `--gpu-memory-utilization`.

## v0.1.17 - 2026-08-24

- Merges [PR #125](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/125), deduplicating named-tool streaming fallback output and preserving `finish_reason=length` for truncated tool calls.
- Merges [PR #128](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/128), disabling implicit repetition detection for tool-call arguments by default so valid repeated Markdown or code is not truncated.
- Merges [PR #129](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/129), emitting Qwen XML function names once per tool call and omitting repeated `name` fields from continuation deltas.
- Merges [PR #130](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/130), making Qwen XML close-tag recovery call-scoped, cross-chunk safe, and bounded for long streamed arguments.
- Merges [PR #132](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/132), bounding the TurboQuant FlashInfer prefill wrapper cache while preserving CUDA Graph-safe wrapper lifetimes.
- Merges [PR #133](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/133), re-aligning Mamba offload hit boundaries after per-group clamps to preserve hybrid KV state consistency.
- Merges [PR #134](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/134), reserving TurboQuant continuation workspace for the configured maximum context before KV-cache sizing.
- Release credit: @kevinhirsch and @weicj.

## v0.1.16 - 2026-08-20

- Merges [PR #101](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/101), bounding Qwen reasoning blocks without dropping split markers.
- Merges [PR #108](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/108), disabling custom all-reduce during CUDA Graph profiling to avoid the SM75 128K IPC leak path.
- Merges [PR #111](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/111), reserving TurboQuant continuation-prefill workspace before KV-cache sizing to avoid high-context Xid 31 crashes.
- Merges [PR #116](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/116), preventing `None` function names in Qwen3 XML streaming tool-call chunks.
- Merges [PR #120](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/120), adding interactive and non-interactive TP/PP layout selection to the 0.1.x launcher with GPU-factorization validation.
- Release credit: @YuYue1208, @superniker, @kevinhirsch, @hotwa, and @weicj.

## v0.1.15 - 2026-08-15

- Validates the official
  [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) checkpoint
  end to end on the dual RTX 2080 Ti TP=2 runtime. The Qwen3.x 27B FP8 route
  now explicitly covers official Qwen3.6 and Qwen3.8 checkpoints.
- Merges [PR #85](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/85),
  improving build download-route preflight, PyTorch fallback
  behavior, and automatic `MAX_JOBS` limits for reliable source builds.
- Merges [PR #89](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/89),
  preserving hybrid Mamba prefix-cache correctness when MTP
  speculative decoding is enabled.
- Merges [PR #91](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/91),
  adding native CPU KV offload support for the Mamba align
  cache path.
- Merges [PR #93](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/93),
  preserving the GDN causal-convolution state slot zero during
  SM75 decode and adding its regression coverage.
- Merges [PR #98](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/98),
  completing named tool-choice response handling for full and
  streaming OpenAI-compatible chat completions, including Mistral-compliant
  IDs and empty-argument handling.
- Release credit: @weicj, @YuYue1208, and @0xYYP.

## v0.1.14 - 2026-07-07

- Merges [PR #78](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/78),
  improving the validated SM75 TurboQuant long-context route with
  the tested continuation prefix-combine path, tuned decode `BLOCK_KV=2`
  defaults, reproducible long-context benchmark controls, and launcher/runtime
  plumbing for the shipped TurboQuant throughput lane.
- Merges [PR #81](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/81),
  finalizing the non-interactive launcher override path with
  explicit `CLI > ENV > PROFILE > default` precedence, `CUDA_VISIBLE_DEVICES`
  mapping, mode-derived override hygiene, and matching English / Simplified
  Chinese launcher documentation.
- Release credit: @0xYYP and @weicj.

## v0.1.13 - 2026-07-04

- Merges [PR #71](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/71),
  fixing the SM75 TurboQuant TQK8V4 FP8 key-format path and the
  launcher submenu numeric-selection regression.
- Merges [PR #72](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/72),
  adding the first validated Docker runtime packaging path for
  the SM75 fork, including the runtime Dockerfile, compose example, entrypoint
  flow, and packaged helper assets needed to launch the shipped profiles
  inside a container.
- Extends the merged Docker path with the validated host-side `docker/build.sh`
  wrapper so Docker builds follow the repository `build.sh` behavior for
  automatic `MAX_JOBS` selection, download-route preflight, and consistent
  PyPI/Git mirror propagation through Docker and compose builds.
- Release credit: @0xYYP and @hotwa.

## v0.1.12 - 2026-07-02

- Integrates the SM75 custom all-reduce graph-input auto policy so the official
  Qwen3.6 35B FP8 release route can keep the validated fast decode path while
  avoiding the PIECEWISE graph-capture startup crash.
- Finalizes the current build/install reliability fixes, including safer build
  parallelism defaults, broken virtualenv self-heal, dependency checkout reuse,
  and rebuild recovery for fresh source installs.
- Adds the validated official Qwen3.6 35B FP8 profile set: 256K text-only
  `normal` and `aggressive`, 136K text+image `normal` and `aggressive`, and a
  178K `fast` MTP3 route.
- Fixes the shipped Qwen3.6 27B `fast` TQK8V4 256K prefix-cache continuation
  route by promoting the validated `GPU_UTIL=0.96` profile values and making
  the launcher auto-reserve the larger continuation workspace for long
  single-sequence TurboQuant lanes.

## v0.1.11 - 2026-06-29

- Updates the launcher defaults for long-context serving: prefix cache is now a
  launcher-level default, prompt token details are enabled for status
  visibility, and Qwen routes automatically use the cache mode required by the
  validated prefix-cache path.
- Adds launcher-side startup safeguards for large profiles, including
  cold-compile prewarm retry and display-GPU occupancy warnings.
- Retunes the shipped Qwen3.6 TQK8V4 `fast` profiles for the validated prefix
  cache path by raising their max batched tokens to `2560`.

## v0.1.10 - 2026-06-17

- Fixes the Qwen reasoning startup path by keeping `reasoning_content`
  compatible with clients that still read the legacy field name.
- Unifies chat reasoning and tool-call parsing so `content=None` responses do
  not break named tool choice handling, and adds regression coverage for the
  reasoning-only null-content path.
- Fixes Qwen GDN mixed prefill/decode handling when mixed batches transition
  through the recurrent attention path, including the TurboQuant attention state
  handling needed by that route.
- Adds the explicit GDN prefill metadata used by the validated mixed
  prefill/decode path, and re-validates the FP8 FP16-KV 128K MTP3 route on
  dual RTX 2080 Ti.

## v0.1.9 - 2026-06-15

- Restores the validated Qwen3.6 27B FP8 + TurboQuant TQK8V4 `fast` route on
  SM75 by pinning the working FlashInfer 0.6.8 runtime path and guarding
  unvalidated newer FlashInfer ragged-prefill behavior.
- Adds TurboQuant launcher defaults for the validated FlashInfer `fa2` prefill,
  workspace reservation, and speculative decode fast path used by the tested
  dual 2080 Ti route.
- Carries FlashQLA legacy SM70/SM75 patch assets into source builds so fresh
  builds reproduce the validated GDN prefill path.
- Improves one-click build reliability with network preflight, package/git
  mirror fallback, local wheelhouse support, CUDA toolkit sanity checks, and
  clearer hardware/environment hints.
- Adds `update.sh` for release-version checks and optional rebuild after an
  update.
- Documents the recommended host runtime as Ubuntu 22.04/24.04 LTS or Debian
  12 on Linux kernel 6.x. Ubuntu 26.04 / CUDA 13 remains a community
  experiment until runtime profiles are validated.

## v0.1.8 - 2026-06-14

- Improves source-build reliability for non-Miniclaw dual 2080 Ti hosts,
  including automatic FlashQLA SM70/SM75 install and clearer runtime path logs.
- Fixes local launch reliability around service stop, failed-start cleanup,
  generated-kernel cache isolation, and default `normal` mode handling.
- Adds an explicit `aggressive` launcher mode: a more aggressive mode with the
  highest performance and quality risk.
- Tightens SM75 backend selection so Turing hosts do not accidentally prefer
  unsuitable FlashAttention paths over the validated FlashInfer/TurboQuant
  routes.
- Carries the SM75 TurboQuant compatibility fixes for workspace reservation,
  sliding-window decode, and legacy runtime environment handling.
- Refines Qwen GDN / linear-attention decode correctness and removes debug-only
  overhead from the normal serving path.
- Updates hardware notes for CPU/platform latency after local build-to-launch
  validation showed old Xeon hosts can underperform newer low-end desktop CPUs.

## v0.1.7 - 2026-06-13

- Updates the `safe`, `normal`, and `fast` launch modes. `normal` is now the
  default recommended mode.
- Fix OpenAI-compatible tool calling, including automatic tool choice and
  structured tool-call output.
- Fixes SM75 TurboQuant compatibility gaps reported against the legacy
  `fast` + INT8-KV route: all ubatch workspace reservation, decode
  `sliding_window` handling, and fork-specific runtime environment variables.
- Improves the current Qwen and Gemma serving routes with updated stability,
  long-context, and profile validation work.
- Includes the Gemma4 checkpoint-name remap fix for official QAT and
  multimodal-style checkpoints.


## v0.1.6 - 2026-06-09

- Adds explicit W8A8 checkpoint support documentation for the Quark INT8 route,
  including the tested `nameistoken/Qwen3.6-27B-Quark-W8A8-INT8` checkpoint.
- Updates the launcher display to separate the real vLLM `--quantization`
  value from the display-only W/A type (`W4A16`, `W8A16`, `W8A8`).
- Adds launcher service-status cache reporting for running services. Live
  `used` values and 30-second refresh are shown only when vLLM exposes real
  cache-usage metrics; otherwise the launcher reports total cache capacity only.
- Improves launcher startup preflight for large single-file checkpoints and
  cleans up residual vLLM processes after failed launches.
- Fixes launcher stop handling so orphaned vLLM API servers, worker processes,
  and resource trackers are discovered and cleaned up instead of leaving VRAM
  occupied.

## v0.1.5 - 2026-06-08

- Renames the public service manager to `launcher.sh` and keeps `build.sh` as
  the one-click source build entry point.
- Updates launcher modes to `safe`, `normal`, and `fast`, with route profiles
  split by model, mode, and weight precision.
- Adds chat template presets and service-level thinking budget defaults while
  keeping global runtime controls out of route profile files.
- Refreshes the Qwen3.6 profile documentation and restores the KV throughput
  sweep SVG charts.

## v0.1.4 - 2026-06-06

- Slims the public repository down to the focused SM75 runtime source tree,
  launcher scripts, validated profiles, and project documentation.
- Keeps Docker artifacts out of this source release; Docker packaging remains a
  separate future deployment layer.
- Adds the interactive `launcher.sh` service manager and one-click `build.sh`
  source build entry point.
- Uses the public launcher modes `safe`, `normal`, and `fast`, with validated
  profiles organized under model-specific profile directories.
- Carries forward the `v0.1.3` graph-safety runtime fixes while removing
  upstream CI/docs/test bulk from the public source tree.

## v0.1.3 - 2026-06-05

- Adds the issue #24 MTP graph-safety fix for hybrid Mamba/GDN models.
- Makes production profiles safer by default: Native MTP + hybrid recurrent KV
  layers fall back from full decode CUDA Graph replay to PIECEWISE/NONE.
- Keeps the old peak-throughput route available for explicit speed benchmarking
  via `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=1`.

## v0.1.2 - 2026-06-04

- Public stable snapshot for the SM75 TP=2 CUDA 12.8 runtime.
- Keeps the upstream vLLM base at `0.21.0` while versioning this fork as an
  independent 2080 Ti runtime distribution.
- Updates the documented Qwen3.6 and Gemma4 runtime routes, tested checkpoint
  list, launcher profile guidance, and benchmark evidence links.

## v0.1.1

- Follow-up compatibility fixes for editable/source bu
