# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## v0.3.0.dev0

- Adds an experimental Docker runtime based on the official vLLM v0.34.0
  release image with the SM75 FlashQLA prefill extension, FlashInfer FP8 KV
  compatibility, and a fixed KV cache default for local Qwen GDN on SM75 TP2.
- Removes the bundled vLLM source tree and native build tooling from 0.3.x.
  The build, launcher, and update commands manage Docker deployments. Native
  source builds are available in the 0.2.x release line.
- Uses FlashInfer for TurboQuant prefill on SM75. Records Docker results for
  the 2xT10 Qwen3.8-27B-INT4 MTP/3 profiles with FP8, TQK8V4 and TQ4NC KV.
- Validates the shipped 4xT10 throughput profiles on the Docker runtime.
  Uses BLHNC pages and batched DFlash verification for TurboQuant and unit
  cache scales for unquantized compressed-tensors KV.
- Groups TP4 Qwen DFlash2 target layers independently of the draft layer
  count to recover usable KV capacity on SM75.
- Includes Inkling 2 Mini support in the default Docker build, using the
  upstream model implementation and lvb-s's SM75 port. Adds pure AR and
  DSpark/7 profiles validated with the official checkpoint on 2xT10 and 4xT10
  over PCIe P2P.
- Adapts Inkling 2 Mini image preprocessing and vision profiling to vLLM
  v0.34.0. Validates DSpark/7 with FP8 KV and two concurrent image requests
  at 12,000 tokens per request on 2xT10 and 34,000 on 4xT10.
- Renames the Qwen3.8-27B and Qwen3.6-35B profile directories to `qwen38-27b`
  and `qwen36-35b`. The launcher resolves the previous paths. The hardware
  guides report the current Docker measurements and test environment.

## v0.2.2-post3 - 2026-10-01

Patch release following `v0.2.2-post2`.

### Optional local SSD prefix KV persistence (#243)

Adds experimental, opt-in prefix KV persistence on local SSD through the
launcher, with checkpoint fingerprints and stable hashes for reuse across
restarts. Cache recovery was validated after a host reboot on 4xT10 and after
service restarts on dual RTX 2080 Ti MTP4 and DFlash2 routes.

### Hybrid Mamba prefix-cache correctness (#241)

Restricts fine-grained Mamba prefix hits to backends that can restore their
recurrent state, including for Mooncake Store; GDN/FlashQLA uses block-aligned
hits while FlashKDA retains supported fine-grained reuse. The growing-prefix
reproducer passed 40/40 checks on both 4xT10 and dual RTX 2080 Ti routes.

### Launcher interaction and GPU topology (#244)

Adds confirmed Esc cancellation during startup, wraparound menu navigation,
separate histories of up to 10 local target and draft directories, and
topology-aware TP grouping and PP stage ordering with P2P checks.

## v0.2.2-post2 - 2026-09-26

Patch release following `v0.2.2-post1`. This release further strengthens
DFlash2 prefix-cache replay and target-state handling.

### DFlash2 prefix cache and target state

- Retains DFlash2 target replay state across prefix-cache reuse and alternating
  prefill/decode requests, so reusable target state is not lost between turns.
- Restores the configured target Mamba speculative state pages for DFlash2 and
  DSpark while keeping draft KV pages in their independent pools.
- Accounts for the resident replay and hashed-boundary pages together with the
  target speculative pages in Mamba align mode, keeping target block tables and
  KV capacity consistent during speculative replay.
- Adds regression coverage for prefix-cache replay and DFlash target state
  sizing and accounting.

## v0.2.2-post1 - 2026-09-26

Patch release following `v0.2.2`, based on the latest `main` runtime.

### DFlash and hybrid Mamba correctness

- Keeps requests that already emitted decode tokens active while a later request
  continues chunked prefill, preventing unnecessary decode pauses.
- Stops reserving speculative Mamba state pages for DFlash2 and DSpark target
  KV, while retaining the resident replay state required by align mode.
- Preserves hashed Mamba align boundary state across decode and releases only
  obsolete unhashed blocks, keeping DFlash target prefix-cache entries reusable.
- Preserves restored-token accounting through the following DFlash decode step.

### Profiles and validation

- Adds four validated 2xT10 Qwen3.8-27B INT4 W4A16 MTP3 routes covering FP8 KV,
  TQ4NC, TQK8V4, text-only, and text-image serving.
- Adds focused scheduler, KV sizing, and prefix-cache regression coverage for
  the repaired paths.

## v0.2.2 - 2026-09-25

Incremental release from the validated `v0.2.1` SM75 serving snapshot
(`b296055efd`) through the current `main` branch. This release focuses on
correctness and reproducibility for Qwen3.8 hybrid Mamba serving on Turing
GPUs, and introduces the published runtime identity `vllm-def-cu130`.

### Prefix caching and speculative decoding

- Reuses DFlash2 target KV across prefix-cache hits and retires replaced
  hashed Mamba states at aligned prefill boundaries.
- Fixes FP8 KV hybrid prefix hits when the Mamba block size equals the hash
  block size, including the 1648-token SM75 route.
- Preserves request-row boundaries and hardens FULL-graph padding during
  speculative replay.
- Keeps SM75 DFlash2 tensor-parallel reductions in one scale domain.

### MTP, GDN, and TurboQuant correctness

- Prevents late prefills from pausing active MTP decode and preserves exact
  verifier KV when a late prefill joins an MTP batch.
- Fixes the hybrid-Mamba prefill barrier livelock and adds explicit alignment
  limits and tail-cohort coverage.
- Stabilizes TurboQuant verifier logits across late prefills and fails closed
  when adaptive query boundaries cannot be honored.
- Partitions stateful Qwen GDN forward execution and fixes SM75 GDN/DFlash
  speculative quality regressions.

### SM75 runtime and launcher

- Prevents unsupported CUTLASS FP8 dispatch on SM75 by selecting the supported
  fallback path.
- Adds safer FULL-graph padding and trace controls for the SM75 runtime.
- Makes the launcher load the model's generation configuration by default.
- Synchronizes renamed validated profile references and refreshes the shipped
- Adds the validated 4xT10 W8A16 FP8KV DFlash2 text-only route at 2x220K.
- Keeps FlashInfer MTP draft metadata correct across metadata rebuilds and
  preserves eager MTP execution for the SM75 compatibility path.

## v0.2.1 - 2026-09-19

Stable release for the CUDA 13 / PyTorch 2.13 SM75 line.

- Promotes the validated SM75 serving routes and profile set from the RC.

## v0.2.1-RC1 - 2026-09-18

Release candidate for the CUDA 13 / PyTorch 2.13 SM75 line.

- Adds validated DFlash2 routes, fused Triton grouped convolution, stable SM75
  FlashInfer FA2 graph buffers, and live metadata refresh for FULL graph replay.
- Fixes concurrent speculative graph sizing, automatic prefill alignment,
  independent target/draft KV pools, and concurrent MTP FA2 execution.
- Generalizes qualified PCIe custom all-reduce to 2–16 GPUs and improves the
  CUDA 13 Torch/Triton mirror build path.
- Converges profiles on validated routes, requires explicit KV precision, and
  removes unsupported concurrency presets.
- Moves profiles to a flat hardware/model/weight route layout and makes `fast`
  the launcher default startup mode.

## v0.2.1-pre4 - 2026-09-14

Nightly-alignment candidate based on upstream `b23433088b`
(`v0.29.1rc0-33`).

- Adopts upstream replacements for previously local runtime fixes and retains
  only the SM75-specific FlashQLA, TurboQuant, parser, and rendezvous paths.
- Keeps `humming-kernels[cu13]==0.1.13` for INT6/AutoRound loading.

## v0.2.1-pre3 - 2026-08-27

Validated true-concurrency candidate for dual RTX 2080 Ti.

- Adds packed-varlen FlashQLA, prefill batch alignment, bounded TurboQuant
  wrapper caching, and long-context workspace handling.
- Improves reasoning/tool parsing, Mamba/offload boundaries, GPU selection, and
  single-node process rendezvous.
- Validates the documented NVFP4 concurrency route and the associated parser,
  serving, TurboQuant, offload, and FlashQLA paths.

## v0.2.1-pre2 - 2026-08-21

First validated preview based on upstream vLLM `v0.27.1`.

- Adds Qwen3.8 FP8/NVFP4 profiles, TP/PP launcher selection, stricter preflight
  checks, and the required SM75 FlashInfer/FlashQLA compatibility paths.
- Ports the validated reasoning, tool-call, CUDA Graph, and TurboQuant fixes to
  the newer runtime architecture.
- Validates the documented dual-2080-Ti TP=2 routes; broader PP and additional
  long-context routes remained experimental.

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based
on upstream vLLM `v0.27.1`.

Thanks to upstream vLLM and its contributors. Fork release and validation work
is maintained by [@weicj](https://github.com/weicj), with contributions from
the project community.
