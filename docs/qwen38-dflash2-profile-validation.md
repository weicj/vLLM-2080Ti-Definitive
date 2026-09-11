# Qwen3.8 DFlash2 Profile Validation

This is the reproducibility record for
`profiles/qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env`.

## Route

- Target: `nvidia/Qwen3.8-27B-NVFP4`
- Draft: `incoai/Qwen3.8-27B-DFlash2`
- Hardware: two RTX 2080 Ti (SM75), P2P enabled, custom all-reduce auto
- KV: TurboQuant K8V4
- Speculation: DFlash2, K=7, `TRITON_ATTN`
- Target graph: `PIECEWISE`, capture size 8
- TQ target verification: causal synthetic sequence lengths, B=8 chunks
- Context: `MAX_MODEL_LEN=262144`
- Scope: text-only, one sequence

## Evidence

The fixed high-acceptance 4K prompt / 128 completion lane produced these
decode rates in three independent runs:

```text
155.164865 tok/s
166.566925 tok/s
166.756140 tok/s
mean 162.829310 tok/s
```

All runs returned 128/128 tokens, HTTP 200, and a normally terminated stream.
The correctness smoke returned `PROFILE_OK` three times. A near-full
`262016`-token prompt completed 16/16 output tokens. Real output-stage stress
tests generated approximately 4K and 8K tokens of an interactive HTML/Canvas
task; the latter included JavaScript and remained healthy through the length
limit. The service stayed alive and its log showed no new CUDA illegal
instruction, `EngineDead`, fatal error, or stream interruption.

The DFlash2 unit suite passed on the target runtime:

```text
17 passed in 10.84s
```

## Reproduction

Use the profile with the normal launcher route:

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
PROFILE=qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env \
GPU_DEVICES=GPU-UUID-0,GPU-UUID-1 TP_SIZE=2 MODE=normal \
bash launcher.sh --non-interactive --print-config
```

The final environment must contain `SPECULATIVE_METHOD=dflash`,
`SPECULATIVE_TOKENS=7`, `SPECULATIVE_MAX_MODEL_LEN=262144`,
`SPECULATIVE_ATTENTION_BACKEND=TRITON_ATTN`,
`VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1`, and
`VLLM_TURBOQUANT_SPEC_DECODE_CHUNK_SIZE=8`.

The B=8 setting is an SM75 performance tuning parameter, not a semantic change:
each target verification row still receives its own progressively increasing
causal sequence length. Do not disable the safe branch when using this profile.
