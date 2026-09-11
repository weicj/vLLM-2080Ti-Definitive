# Changelog

## Unreleased

### Removed

- Remove the plugin-owned `vllm_exl3_c` CUDA extension, including native
  p2b MoE and fat-GEMM dispatch. All EXL3 compute now runs through the
  maintained ExLlamaV3 runtime kernels.

### Fixed

- Dense EXL3 calls no longer launch exllamav3's cooperative trellis GEMM. Rows 17 to 144 take exllamav3's reconstruct+hgemm path (exact); rows up to 16 keep exllamav3's own dispatch (GEMV up to 2 rows, cooperative GEMM for 3 to 16 rows, which ran clean through the same 45-minute stress and where reconstruct would cost about 10x per call). On the vLLM nightly V2 model runner the cooperative GEMM wedged the engine (EngineCore at 100% CPU, GPU busy at idle power): deterministically on 33 to 144-token prefills and after 30 to 60 minutes of MTP k=2 decoding. A 4-worker MTP stress that wedged within 3 minutes ran clean for 45 minutes (1085 requests, 77 tok/s aggregate) once the routing was in place, with decode speed unchanged and TTFT on 32-token prompts about 0.25 s. `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch; `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the reconstruct threshold. `VLLM_EXL3_PREFILL_SYNC` is no longer needed for this and stays as an optional knob.

- Fat-expert prefill path (`apply_exl3_batched_fat`, experts with more than `VLLM_EXL3_FAT_THRESHOLD` routed rows in a chunk): the branch for packs whose gate and up projections carry distinct `suh` rotations handed column slices of the shared `gate_up` scratch buffer to `ext.hgemm` and `ext.had_r_128`. Those kernels index contiguous row-major operands, so the expert output was uncorrelated with the reference (relative error 1.3, cosine 0.01 on a Qwen3.8-Flash-Next expert) while short prompts, which never reach that path, looked normal. Prompt log-likelihood over a 6000-token corpus was mean NLL 4.21 through vLLM against 0.94 through exllamav3 on the same pack; with the fix it is 0.943. The branch now runs on contiguous fp32 temporaries. Packs with a shared gate/up `suh` (fused gate_up quantization) were not affected. Regression test: `tests/test_fat_distinct_suh.py`.

- Add `Exl3EmbeddingMethod` for row-wise n-gram embedding tables
  (`ngram_embedding`), decoded through the compiled
  `exllamav3_ext.ngram_dequant` kernel or a pure-torch fallback
  (`VLLM_EXL3_NGRAM_KERNEL=ext|torch`).
- Add the `ngram_embedding` config spec (`bits`, `num_shards`,
  `rows_per_shard`, `num_heads`, `modules`) and its checkpoint layout
  (`shard_<i>.trellis`, `head_bias`, `head_offsets`, `head_vocab_sizes`,
  `layer_multipliers`); tensor parallel size 1 only.
- Extend `get_quant_method` with branches for `ParallelLMHead` and
  `VocabParallelEmbedding`, so EXL3 `lm_head` and n-gram tables resolve
  once a model passes `quant_config=` through to those layers.
- Accept tuple and `None` shard-id spans in the dense linear weight
  loader, for fused modules with more than one contiguous shard and for
  full-tensor (non-sharded) loads.
- Generalize `_check_moe_codebook_markers` to validate `mul1` markers
  for routed experts alongside `mcg`.
- Add `_exl3_routed_experts_loader`, a per-expert `load_weights` path for
  routed-experts layers that mirrors vLLM's own checkpoint-name
  resolution for one-tensor-per-expert EXL3 checkpoints.
- Pad dense EXL3 linear geometry to multiples of 128 so trellis tiles
  never spill past the real matrix dimension.
- Thread per-tensor codebook flags (mcg/mul1) through the fused
  `exl3_moe` launch arguments.
- Register the EXL3 custom ops so they trace opaquely under
  `torch.compile(fullgraph=True)` instead of breaking graph capture.
- Restore the CUDA-graph-safe fat-expert-sync guard (skip the device
  sync entirely when the row count cannot produce a fat expert) and the
  MTP/draft-expert quant-method delegate (prefer MXFP4, matching DSV4's
  own fp4 draft experts, before falling back to the non-routed delegate).
- Add `tools/patch_vllm_qwen4_exp` (vLLM quant_config plumbing for
  `Qwen4ExpForConditionalGeneration`), `tools/exl3_pack_tools`
  (pack scanning, config rewriting, and
  `regenerate_safetensors_index.py`), and `tools/verify_native_pack`
  (four pre-boot GPU correctness gates).
- Add attribution notices for ExLlamaV3's n-gram embedding codec and for
  vLLM's routed-experts loader / custom-op registration; thank
  turboderp for the Qwen3.8-Flash-Next-exl3 pack used to validate this
  release.

### Added

- `VLLM_EXL3_PREFILL_SYNC=<max_rows>`: synchronizes the device before each EXL3
  dense or routed-expert call whose row count is between 2 and max_rows (never
  during CUDA graph capture, never for single-row decode). Workaround for a vLLM
  nightly V2 model runner wedge on Qwen3.8-Flash-Next where prefills of roughly
  33 to 144 tokens never complete (EngineCore at 100% CPU, GPU busy at idle
  power); the same request completes with the synchronizations in place, and
  decode speed is unchanged. Recommended value 256 on that runner.

## 0.3.1

- **Super Fat GEMM Prefill Kernel Suite (`csrc/exl3_fat_gemm.cu`, `csrc/exl3_fat_gemm.cuh`)**:
  - Tiled chunked prefill kernel optimized for wide-layer routed expert evaluation during high-context and large-batch prompts.
  - Implements batched matrix multiplication over unquantized and trellis-dequantized states with register-level unrolling.
  - Dispatched automatically via `apply_exl3_batched_fat` in `src/vllm_exl3/exl3.py`.
- **Bug Fix**:
  - Guard `k == 4` in `apply_exl3_batched_fat` dispatch to prevent illegal memory layout indexing when handling 4-bit trellis tiles.
- **Upstream Attribution & Notice Compliance**:
  - Full third-party attribution prominently placed at the top of `README.md` and detailed in `THIRD_PARTY_NOTICES.md`.
  - Credits to @MiaAI-Lab and @plotarmordev for the routed-expert EXL3 serving path and Fat GEMM CUDA kernels (`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, commit `4b8d3c7`).
  - Credits to @turboderp for the ExLlamaV3 trellis quantization format, MCG codebook, and base dequantization math.
- **Hardware Benchmarks**:
  - Benchmarked on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory.
- **Dynamic Speculative Draft Scheduler**:
  - `get_speculative_draft_tokens` selects K dynamically by batch size: `[1..4]` → 3, `[5..8]` → 2, `[9..16]` → 1, and larger batches → 0.
  - `VLLM_EXL3_SPEC_SCHEDULE` provides a validated `min:max:k` override.
- **Vectorized On-Device Confidence Pruning**:
  - `filter_speculative_candidates` truncates each candidate stream at its first below-threshold confidence without host-side loops.
  - `VLLM_EXL3_ADAPTIVE_VERIFICATION` enables the opt-in verification path.
- **Context Ceiling Scaling & MLA KV Cache Headroom**:
  - `compute_mla_kv_cache_bytes` and `validate_context_scaling` cover 64K (1.51 GiB), 128K (3.02 GiB), and 256K (6.05 GiB) FP8 MLA KV storage.

## 0.3.0

- **Native EXL3 CUDA Kernel Suite (`csrc/`)**: High-performance native CUDA kernels replacing `exllamav3_ext` decode and prefill paths on NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory:
  - **In-Register Trellis Dequantization (`csrc/exl3_dequant.cuh`)**: Unrolls MCG bit extraction into hardware registers without intermediate global memory roundtrips.
  - **Active-Expert Batched GEMV (`csrc/exl3_gemv.cu`, `csrc/p2b_batched.cu`)**: Saturates 99.2% of the physical memory bandwidth floor (73.3 μs).
  - **4-Phase Cooperative MoE Decode (`csrc/p2b_moe.cu`)**: End-to-end fused MoE decode reducing per-layer latency from 497 μs → 287.8 μs (1.73x speedup).
  - **Power-of-Two Chunked Prefill GEMM (`csrc/exl3_gemm.cu`)**: Tiled matrix multiplication delivering 7.85 TFLOPS (13.0x faster than legacy prefill).
  - **vLLM Dispatch Control**: Environmental toggle `VLLM_EXL3_MOE_KERNEL=native` (default) with zero-cost fallback to `exllamav3`.
- **NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory Live Benchmark Receipts**:
  - Measured across NVIDIA DGX Spark GB10 (sm_121 Blackwell) with 128 GiB Unified Memory nodes running GLM-5.3-Flash EXL3 K2 via live vLLM HTTP streaming API:
    - Coding: 14.9 tok/s → 27.6 tok/s (+85.6% speedup, TTFT 2,343.8 ms → 859.1 ms)
    - Prose: 13.7 tok/s → 24.6 tok/s (+79.3% speedup)
    - Summary: 17.1 tok/s → 25.6 tok/s (+50.0% speedup)
    - Average Across Categories: 16.9 tok/s → 24.6 tok/s (+45.6% speedup)
  - MoE Compute: Cut from 19.9 ms → 11.5 ms per token (-42.2%), saving 8.4 ms in pure MoE decode compute.
  - Prefill: 1,875 tok/s cold prefill across 65k context.
- **Serving Guidance**:
  - Added `--long-prefill-token-threshold 1024` recommendation to prevent long prompt prefill from starving parallel decode steps.
- **Attribution & Notice Compliance**:
  - Full third-party attribution and notices for Turboderp (@turboderp) and Mia's AI Lab (@MiaAI-Lab) documented in `THIRD_PARTY_NOTICES.md`.

## 0.2.3

- Dense EXL3 for non-routed linears (`quantization_config.non_routed_exl3`): per-module `layers` map with `bits` and `bf16_shards`, `mul1` codebook alongside `mcg`, mixed EXL3/BF16 shards inside one merged linear, stale BF16 `.weight` tensors discarded with a shape check. `tools/dense_overlay.py` assembles an overlay pack from an existing EXL3 checkpoint (it moves tensors, it does not quantize).
- `quantization_config.non_routed_dtype_policy: "bf16_as_stored"`: dense linears go to vLLM's unquantized method instead of the `non_routed_quantization` delegate, which still serves source-format MTP experts. Fixes silent empty output when BF16 dense weights met an fp8 delegate (DeepSeek-V4-Flash on stock vLLM 0.28).
- Fix: `mul1` codebook marker constant (0x83DCD12D as signed int32 is -2082680531).

## 0.2.2

- Mixed-format packs: `quantization_config.mtp_experts: "source"` routes MTP/draft-block routed experts through the declared `non_routed_quantization` method (e.g. MXFP4) instead of EXL3, enabling MTP speculative serving for packs that keep drafter experts in the source format. Default (`"exl3"`) is unchanged.

## 0.2.1

- Fix: the `glm53_exl3_plugin` compatibility shim now provides a real `glm53_exl3_plugin.exl3` submodule (submodule imports bypassed the lazy alias).

## 0.2.0

First standalone release. Renamed from `glm53_exl3_plugin` 0.1.1 — identical
behavior plus the package rename; the old import path remains as a
deprecated shim.

## 0.1.1 (as glm53_exl3_plugin, shipped in the GLM recipe)

- Raise the fused-MoE per-expert row cap (`TEMP_ROWS_FUSED` 128 → 2048),
  fixing the >163k-token prefill stall where fat experts fell back to a
  slow per-expert reconstruction path.
- Delegate non-routed layers to a pack-declared source-format quant method
  (`quantization_config.non_routed_quantization`).

## 0.1.0 (as glm53_exl3_plugin)

Initial in-recipe release: EXL3/MCG routed-expert quantization method for
vLLM fork runtimes, per-layer `layer_bits` support.
