# 4xT10 Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env` | 240K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
| `qwen27b/w8a16/dflash2-fp8kv-2x220K-text-only.env` | 2 x 220K | FP8 | DFlash2/7 | text-only | 445,644 | 1395.16 / 200.56 | 1486.57 / 210.18 |
| `qwen27b/w8a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 320,484 | 1484.30 / 106.04 | 1527.71 / 104.25 |
| `qwen27b/w8a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 565,524 | 1459.21 / 102.41 | 1521.55 / 104.32 |

## Qwen3.8-27B-NVFP4

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 432,205 | 1435.68 / 220.99 | 1505.42 / 216.84 |
| `qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env` | 2 x 262K | FP8 | DFlash2/7 | text-only | 763,494 | 1334.35 / 213.94 | 1348.40 / 228.76 |
| `qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env` | 4 x 220K | TQK8V4 | DFlash2/7 | text-only | 923,137 | 1374.96 / 234.92 | 1380.19 / 154.43 |
| `qwen27b/w4a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 436,388 | 1526.32 / 125.70 | 1517.31 / 123.13 |
| `qwen27b/w4a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 823,249 | 1507.94 / 116.86 | 1477.49 / 118.77 |

## 0.3.x Docker validation

These tests use the official vLLM 0.34.0 image with the project's
SM75 patches, four PCIe P2P-capable T10s, FP16 activations and 2,048 batched
tokens. Throughput uses the same text request protocol as the tables above.

| Profile | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env` | 331,998 | 1583.68 / 211.09 | 1466.17 / 210.14 |
| `qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env` | 293,387 | 1584.73 / 209.94 | 1443.93 / 218.32 |
| `qwen27b/w8a16/dflash2-fp8kv-2x220K-text-only.env` | 515,625 | 1585.14 / 211.10 | 1344.57 / 209.08 |
| `qwen27b/w8a16/mtp4-fp16kv-1x262K-text-image.env` | 345,301 | 1578.10 / 121.59 | 1454.63 / 120.15 |
| `qwen27b/w8a16/mtp4-fp8kv-2x262K-text-image.env` | 650,056 | 1567.30 / 124.74 | 1428.35 / 125.80 |
| `qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env` | 443,917 | 1559.39 / 249.82 | 1443.68 / 246.30 |
| `qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env` | 703,905 | 1561.66 / 247.63 | 1428.82 / 243.43 |
| `qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env` | 956,153 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| `qwen27b/w4a16/mtp4-fp16kv-1x262K-text-image.env` | 451,426 | 1557.56 / 143.11 | 1437.54 / 137.96 |
| `qwen27b/w4a16/mtp4-fp8kv-2x262K-text-image.env` | 853,104 | 1547.72 / 141.89 | 1412.98 / 124.51 |

The throughput runs use fixed KV allocation, the LBNHC layout except for
4×220K, FlashInfer PCIe IPC all-reduce and launcher mode `fast`. With Prefix
Cache disabled for those runs, GPU utilization was 0.96 for the 240K route
and 0.94 for the other LBNHC routes. The 240K and 2×220K DFlash2 routes
capture CUDA graphs only for decode. All five multimodal routes identified
the blue square and orange circle in the image smoke test.

For DFlash2 with LBNHC pages, the image groups Qwen's target attention and
recurrent layers without padding them to the draft layer count.

With Prefix Cache enabled, the launcher uses `GPU_UTIL=0.965` for 240K and
`GPU_UTIL=0.948` with NCCL over PCIe P2P for 2×220K. Those runs reported
295,117 and 508,888 GPU KV tokens, respectively. Repeated requests hit 3,328 of
4,096 prompt tokens and generated identical token IDs. The 2×220K route
also completed two simultaneous 32K/512 requests with the cache enabled.

The 4×220K route uses BLHNC pages at `GPU_UTIL=0.93`. FlashInfer
shares the dequantized prefix across DFlash verification tokens on SM75.
Four simultaneous 32K/512 requests completed with exact token counts.

The NVFP4 checkpoint uses FP8 weights in part of the model. On Turing the
image uses Marlin for those layers as well as the NVFP4 projections.
For the FP16 KV route, the image uses unit KV scales.

## Notes

1. Profiles are flat under each model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens. Raw logs and request JSON were retained for verification.
3. Test environment: the FP8/W8A16 profiles were validated on 2026-09-19 and the NVFP4/W4A16 profiles on 2026-09-23, both with v0.2.1. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The service uses TP4 across four Tesla T10 GPUs with 16,384 MiB each, connected by PCIe PIX on the same NUMA node. The host driver is NVIDIA 595.91.07. The runtime is the repository's `vllm-def-cu130` environment (CUDA 13.0, torch 2.13.0+cu130).
4. The `4x220K` route uses `GPU_UTIL=0.93` to reserve runtime headroom. Four simultaneous 32K requests passed; four simultaneous requests near the 220K limit are not a supported worst-case guarantee and can OOM on DFlash temporary buffers. Multi-concurrency profiles should not receive simultaneous ultra-long requests.
5. All profiles use the launcher's default `MAX_BATCHED_TOKENS=2048`; enable Prefix Cache in production for FP8KV routes. The measurements above deliberately disabled Prefix Cache. The W8A16 FP8KV DFlash2 2x220K route was capacity-sized from 445,644 GPU KV tokens at `GPU_UTIL=0.96`; two simultaneous 32K/512 requests completed with exact token counts and the service remained healthy. Under that concurrent load, decode throughput was 62.88 and 65.70 tok/s per request.
