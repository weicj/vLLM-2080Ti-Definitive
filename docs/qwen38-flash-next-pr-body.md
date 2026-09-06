## Summary

This PR adds experimental Qwen3.8 Flash-Next support to the `0.2.x` branch.
It covers the Qwen4-Exp model architecture, RadixArk ModelOpt NVFP4 weights on
SM75 through the Marlin W4A16 fallback, and PLE CPU/SSD offload with pipeline
parallel execution.

Two previously validated eight-GPU profiles are included:

- `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env`
- `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env`

Both profiles use FP16 KV, no MTP, text-only serving, chunked prefill with
`max_num_batched_tokens=512`, and non-eager CUDA Graph execution. TP2xPP2 is
not included because it has not passed the same validation gate.

## Changes

- Register the Qwen4-Exp configuration and model implementation used by
  Qwen3.8 Flash-Next.
- Load ModelOpt NVFP4 checkpoints on SM75 through the Marlin W4A16 fallback.
- Add PLE offload support for safetensors-backed CPU/SSD execution.
- Isolate PLE model discovery from the GPU pipeline layer partition.
- Defunctionalize `ple_offload_wait` before PyTorch 2.13 Inductor lowering.
- Allow experimental profiles to provide TP/PP sizes and an explicit PP layer
  partition while keeping GPU selection under launcher control.
- Validate that a profile PP partition contains exactly `PP_SIZE` entries.
- Retune pre-Ampere QSA sparse prefill launches to `BLOCK_N=16` with four
  warps, following upstream QSA issue #441 and merged fix #469. The capability
  check uses the QSA tensor's device so mixed-GPU workers do not inherit device
  0's architecture.

## Validation

Test host:

- Eight Tesla T10 16 GiB GPUs: physical devices `0,2,3,4,6,7,8,9`
- CUDA 13.0, PyTorch 2.13, vLLM 0.27.1 base
- RadixArk/Qwen3.8-Flash-Next-NVFP4
- ModelOpt NVFP4 -> Marlin W4A16, FP16 KV, SSD-backed PLE

Focused tests:

```text
6 passed, 14 warnings in 3.98s
```

The test set covers the PLE offload layer, PP partition isolation in the PLE
worker, and Qwen4-Exp PLE short convolution. Profile validation also passes for
all 14 shipped profiles.

The pre-Ampere QSA launch-profile and FP16 numerical tests pass `10/10` on a
T10 (SM75). A direct QSA probe for 256-wide heads completed the 64, 256, 512,
and 2048-row prefill regimes without Triton `OutOfResources`.

The clean PR tree completed eight-T10 TP4xPP2 model loading, PLE registration,
PyTorch compilation, non-eager `FULL_AND_PIECEWISE` CUDA Graph capture, HTTP
startup, and `/health=200`. The run used the exact shipped profile with
`VLLM_DISABLE_COMPILE_CACHE=1`, so it also exercises a clean compilation path.
This verifies that the earlier PLE PP-partition error and
`auto_functionalized was not removed` Inductor assertion no longer occur.

The same clean run reached the first real 4K/128 request and JIT-compiled the
QSA paged-attention kernels. The request did not produce a complete streaming
sample after that point, so it is not reported as a new throughput result. The
previously completed runtime measurements below remain the only throughput
references for this profile family.

Previously completed fixed-token 4K/128 measurements with the same profile
parameters were:

| Profile | GPU KV tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| TP4xPP2, util 0.92 | 121,139 | 1,615.30 | 21.99 |
| TP4xPP2, util 0.94 | 138,797 | 1,611.20 | 22.17 |
| TP2xPP4, util 0.90 | 131,872 | 1,979.79 | 10.74 |

The util 0.94 TP4xPP2 run completed all three requests, but the first real
request reported a transient allocator OOM while compiling MRoPE. It is
therefore retained as an upper-bound measurement, not as the clean shipped
profile; util 0.92 is the conservative default.

TP2xPP4 capacity probes on the clean PR tree also completed non-eager CUDA
Graph capture and returned `/health=200`: util 0.92 exposed 126,267 GPU KV
tokens, util 0.94 exposed 161,684, and util 0.96 exposed 174,495. Neither
startup reported a CUDA OOM.
Those measurements predate the QSA pre-Ampere dispatch correction and are
retained as historical capacity/startup evidence rather than new throughput
samples. The shipped TP2xPP4 profile therefore remains at the fully measured
util 0.90 setting.

## Limitations

- Both profiles are experimental eight-T10 engineering routes, not replacements
  for the primary dual-2080-Ti TP2 deployment profiles.
- Only no-MTP text serving is included. MTP and multimodal serving are outside
  this PR's validated scope.
- Cold startup is long because the 206-shard checkpoint, PLE discovery, and
  first graph/kernel compilation must complete on every worker.
- The clean PR startup is validated end to end through CUDA Graph capture and
  HTTP health. A post-start 4K/128 streaming request still needs a stable
  repeatable sample after the QSA correction; the current run reaches QSA JIT
  but does not complete the stream, so no unsupported performance claim is
  made here.

## Duplication

This is a fork-specific integration PR for the maintained `0.2.x` branch. It
combines the architecture registration with the SM75 Marlin fallback, PLE
offload fixes, launcher integration, and validated topology profiles; it is not
a duplicate of an upstream model-registration-only change.
