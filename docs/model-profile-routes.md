# Model Profile Routes

This document defines the evidence rules for the 0.2.x deployment profiles. The current route catalog is maintained in [profiles/README.md](../profiles/README.md), and migration evidence is recorded in [the 0.2.x validation report](2080ti-0.2.1-pre-validation.md).

`Profile` is the relative `.env` route selected by `launcher.sh`; the checkpoint, GPU selection, port, chat template, and reasoning defaults remain launcher-level settings.

## Evidence Rule

A promoted route needs a real request that returns HTTP 200, completes its stream, and passes the documented quality smoke. Load-only, health-only, READY-only, empty-stream, or tiny smoke results are not capacity or throughput evidence.

Every benchmark record must identify the exact checkpoint, weight precision, KV precision, MTP setting, context length, graph mode, TP/PP layout, and measurement method. Historical 0.1.x CUDA 12.8 / PyTorch 2.11 numbers are compatibility evidence, not 0.2.x CUDA 13.0 promotion evidence.

## Route Policy

- `normal` is the default production route after the corresponding 0.2.x validation evidence passes.
- `fast` is reserved for routes that pass quality smoke and non-eager CUDA Graph validation.
- `safe` is a diagnosis or compatibility fallback and is not a performance recommendation.
- FP16/default KV is the quality reference; INT8 and TurboQuant KV routes require their own quality and capacity evidence.
- Experimental or unvalidated routes must remain documented as such and must not be promoted by profile naming alone.

For [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
the experimental TP2xPP4 route has a heterogeneous six-T10/two-RTX-2080-Ti
startup/capacity reference of 152,492 GPU KV tokens at `GPU_UTIL=0.92`.
The record uses NVFP4 weights, FP16 KV, MTP=0, a 100K context, non-eager CUDA
Graph, and a startup/health/real-request capacity probe (not a throughput
benchmark).
