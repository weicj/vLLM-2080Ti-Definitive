# Changelog

This changelog tracks releases of vLLM 2080 Ti Definitive Edition separately from upstream vLLM releases.

## Unreleased - Qwen3.8 Flash-Next support

- Adds Qwen4-Exp model/config registration and Qwen3.8 Flash-Next NVFP4 loading
  through the SM75 Marlin W4A16 fallback.
- Hardens PLE offload model discovery for pipeline-parallel workers and removes
  the `ple_offload_wait` functionalization wrapper that breaks PyTorch 2.13
  Inductor graph lowering.
- Adds two experimental eight-T10 profiles: TP4xPP2 and TP2xPP4, with FP16 KV,
  chunk size 512, non-eager CUDA Graph execution, and SSD PLE offload.

## v0.2.1-pre3 - 2026-08-27

`v0.2.1-pre3` is the validated true-concurrency candidate for the CUDA 13 based 0.2.x line. The maintained 0.1.x line remains the project's primary stable release line.

### Changes Since v0.2.1-pre2

- Ports the validated named-tool truncation contract from 0.1.x: incomplete tool arguments preserve `finish_reason=length` in both streaming and non-streaming chat responses.
- Aligns Qwen3 XML streaming with the one-name-per-tool-call contract while retaining the parser-engine implementation used by 0.2.x.
- Bounds the TurboQuant FlashInfer prefill wrapper plan cache with LRU eviction and keeps eviction disabled for CUDA Graph-safe wrappers to avoid dangling captured-buffer references.
- Re-aligns Mamba offload hit boundaries after per-group chunk clamps so hybrid recurrent state cannot extend beyond the reported attention prefix.
- Retains the existing 0.2.x `max_model_len`-aware TurboQuant continuation workspace reservation, which is already the current-architecture equivalent of the 0.1.x #134 fix.
- Adds packed-varlen FlashQLA execution for SM70/SM75 Qwen GDN prefills, mapping concurrent sequences onto the CUDA grid batch axis instead of falling back to a serial or unsupported path.
- Makes the packed-varlen ABI a build/runtime gate: patch application and `build.sh` require both `gdn_forward` and `gdn_forward_varlen`, and an explicitly requested legacy backend fails early if an old `.so` is still present.
- Adds the opt-in `--prefill-batch-barrier` scheduler mode. Peer requests advance on a shared prefill frontier and submit their final prefill together, so the first generated token no longer escapes before the rest of the cohort enters decode.
- Adds the validated `qwen27b/w4a16/normal/fp8kv-16K-nomtp-concurrent.env` profile and fixes route-profile propagation of `DISABLE_PREFIX_CACHING`.
- Resolves numeric launcher GPU selections through `nvidia-smi` to physical UUIDs,
  preventing mixed T10/RTX 2080 Ti ordinal mismatches, and uses the bounded
  no-MTP decode graph ladder `[1,2,4,8,16]` for the shipped eight-sequence route.
- Reuses the caller-owned Qwen GDN pure-prefill output buffer and plans
  TurboQuant mixed first-chunk FlashInfer wrappers per request, covering the
  decode-plus-first-chunk batch shape used by the concurrency route.
- Pins `humming-kernels[cu13]==0.1.13` for the INT6/AutoRound loading fix
  reported in issue #112; the full Minachist checkpoint remains unverified.
- Uses a collision-free `file://` rendezvous for single-node multiprocessing and uniprocess executors, eliminating startup port races while retaining TCP rendezvous for multi-node and data-parallel groups.
- Evaluates all changes merged by stable `v0.1.17`: #125, #129, #132, and #133 are migrated in the current architecture; #128 and #130 are absorbed by the newer sampling/parser paths; #134 is already covered by the 0.2.x workspace reservation.

### Validation

- On the physical dual RTX 2080 Ti TP=2 target, Qwen3.8-27B-NVFP4 with no MTP, non-eager CUDA Graphs, FP8 KV, prefix caching disabled, and exact 4K/128 requests measured strict full-window aggregate decode of `41.53`, `79.94`, `151.19`, and `269.30` tok/s at concurrency 1/2/4/8. C8 is 6.48x C1 and its first-token spread is 1.447 ms.
- The strict aggregate window is all completion tokens divided by the interval from the first request's first token to the last request's completion. At C8 it differs from the post-all-first-token tail metric by only 0.04%.
- Packed-varlen FlashQLA matches the per-sequence CUDA reference exactly (`out_max_abs=0.0`, `state_max_abs=0.0`). Focused concurrency tests pass 3/3; the `v0.1.17` parser, serving, TurboQuant, and offload regression set passes 301 tests with 17 environment skips, and explicit repetition-detection coverage passes 19/19.
- The file-store regression suite passes 3/3. On `.31`, a clean dual-RTX-2080-Ti TP2 service initialized both NCCL workers through the same `file:///tmp/vllm_dist_*` rendezvous, served a generation request successfully with `enforce_eager=False`, and shut down without leftover workers or listeners.

## v0.2.1-pre2 - 2026-08-21

`v0.2.1-pre2` is the second public preview of the CUDA 13 based 0.2.x line. The maintained 0.1.x line remains the project's primary stable release line; 0.2.x is intended for testing the newer upstream runtime and must not yet be treated as a replacement.

### Changes Since v0.2.1-pre

- Moves the migration snapshot to an evidence-backed preview based on upstream vLLM `v0.27.1`, CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 source builds.
- Adds profile-driven Qwen3.8 FP8 and NVFP4 routes, an improved interactive launcher, explicit TP/PP layout selection, physical GPU ordering, and stricter build/runtime preflight checks.
- Ports the validated SM75 fixes for Qwen reasoning, named and streaming tool calls, CUDA Graph profiling, TurboQuant workspace reservation, and TurboQuant decode diagnostics to the newer runtime architecture.
- Restores the FlashInfer and FlashQLA legacy paths required by the validated SM75 runtime without importing TileLang during normal model initialization.
- Adds bilingual profile documentation, a validation report, and a [PR migration audit](docs/0.2.x-pr-migration-audit.md) that separates migrated, absorbed, experimental, and unsupported work.

### Validation

- The official Qwen3.8-27B-FP8 checkpoint passed non-eager TP=2/PP=1 CUDA Graph startup and output checks on dual RTX 2080 Ti with FP16 KV and TurboQuant K8V4 routes.
- In the controlled 4K/128 FP16-KV comparison, pre2 measured `1570.57 / 30.01` prefill/decode tok/s without MTP and `1522.72 / 75.83` with MTP3. The corresponding 0.1.x baseline measurements were `1420.07 / 30.01` and `1453.46 / 60.70`.
- Profile validation, GDN slot handling, Mamba align, CPU KV offload, reasoning, tool-call, and serving regressions passed for the documented paths.

### Known Limitations

- Promoted serving support is limited to the documented TP=2/PP=1 profiles. PP greater than one, PP with MTP, and PP with DSpark/DFlash/EAGLE3 are not promoted in pre2.
- TurboQuant long-context prefix-combine and retained Qwen3.x 35B profiles require additional CUDA 13 target-GPU validation before promotion.
- MXFP4 is not supported on SM75 by the current upstream Marlin kernels; NVFP4 and MXFP4 are different formats.

### Credits

Thanks to upstream vLLM and its contributors. Fork release and validation work is maintained by [@weicj](https://github.com/weicj), with contributions migrated or evaluated from [@YuYue1208](https://github.com/YuYue1208), [@superniker](https://github.com/superniker), [@kevinhirsch](https://github.com/kevinhirsch), [@hotwa](https://github.com/hotwa), and [@0xYYP](https://github.com/0xYYP).

## v0.2.1-pre

Initial CUDA 13.0, PyTorch 2.13, Python 3.12, and SM75 migration snapshot based on upstream vLLM `v0.27.1`.
