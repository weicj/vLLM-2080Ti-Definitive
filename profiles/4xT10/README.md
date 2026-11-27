# 4xT10 Profiles

Language: English | [简体中文](README.zh-CN.md)

## Qwen3.8-27B-FP8

Checkpoint [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

DFlash2 draft checkpoint [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env`](qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 331,998 | 1583.68 / 211.09 | 1466.17 / 210.14 |
| [`qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env`](qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env) | 240,384 | FP16 | DFlash2/7 | text+image | 293,387 | 1584.73 / 209.94 | 1443.93 / 218.32 |
| [`qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env`](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 1585.14 / 211.10 | 1344.57 / 209.08 |
| [`qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env`](qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 345,301 | 1578.10 / 121.59 | 1454.63 / 120.15 |
| [`qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env`](qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 650,056 | 1567.30 / 124.74 | 1428.35 / 125.80 |

## Qwen3.8-27B-NVFP4

Checkpoint [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

DFlash2 draft checkpoint [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env`](qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 443,917 | 1559.39 / 249.82 | 1443.68 / 246.30 |
| [`qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env`](qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env) | 2 × 262,144 | FP8 | DFlash2/7 | text-only | 703,905 | 1561.66 / 247.63 | 1428.82 / 243.43 |
| [`qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env`](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| [`qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env`](qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 451,426 | 1557.56 / 143.11 | 1437.54 / 137.96 |
| [`qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env`](qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 853,104 | 1547.72 / 141.89 | 1412.98 / 124.51 |

## Inkling 2 Mini

Official checkpoint [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

DSpark/7 loads `mtp.safetensors` from the official checkpoint and uses probabilistic draft sampling.

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens |
| --- | ---: | --- | --- | --- | ---: |
| [`inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | AR | text-only | 24,503 |
| [`inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | DSpark/7 | text-only | 23,106 |

| Prompt | AR 4K/128 prefill / decode | DSpark/7 4K/128 prefill / decode | Draft acceptance |
| --- | ---: | ---: | ---: |
| Prose | 4314.14 / 77.71 | 4172.03 / 255.02 | 81.2–86.5% |
| Code | 4312.15 / 77.37 | 4184.09 / 251.59 | 86.5% |
| Repetitive | 4192.41 / 76.68 | 4089.48 / 353.55 | 78.6–100% |

### DSpark/7 with text and images

| Profile | Concurrent context | GPU KV tokens | KV memory per GPU | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| [inkling-2-mini/official/dspark7-fp8kv-2x34K-text-image.env](inkling-2-mini/official/dspark7-fp8kv-2x34K-text-image.env) | 2 × 34,000 | 85,298 | 8.22 GiB | 3706.03 / 108.07 | 3745.18 / 117.89 |

The capacity test sent two simultaneous requests, each with one 480 × 480 image, 33,872 input tokens including the image and chat framing, and 128 output tokens. Server metrics showed both requests running simultaneously with zero preemptions. A separate two-image request identified the blue square, orange circle, green square, and purple circle.

The image runs used the default Docker build, FP8 KV, fast mode, and NCCL over PCIe P2P. Vision memory profiling was enabled and Prefix Cache was disabled. GPU memory utilization was 0.94, with a 2,048-token batch limit. Throughput is in tok/s for one streamed multimodal request at a time. The 4K/128 figures are medians of three requests and 32K/512 used one request. vLLM calculates equivalent KV capacity at the configured context and reported 2.51 full-context requests at startup.

With a 1,024-token batch limit, four 34,000-token requests completed. Server metrics showed at most three running simultaneously, with the last request waiting for scheduling.

## Capacity and feature checks

The Qwen throughput runs use fixed KV allocation, the LBNHC layout except for
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

The 2×262K NVFP4 DFlash2 FP8 KV route completed two concurrent 32K/512 requests. DFlash temporary buffers can exhaust VRAM when multiple requests approach the configured context limit together.

## Test version and environment

- 2026-11-27, vLLM 2080 Ti Definitive Edition `v0.3.0.dev0`. The image uses official vLLM `v0.34.0`, CUDA 13.0, PyTorch `2.14.0+cu130`, and NVIDIA driver 610.43.03.
- 4 Tesla T10 GPUs with 16 GiB each, TP4, and PCIe P2P within one NUMA node. Tests used `fast` mode. Text-only throughput runs used a 2,048-token batch limit. Inkling used NCCL PCIe P2P.
- Throughput is in tok/s. For the text-only throughput tables, Prefix Cache was disabled and requests ran one at a time after warmup. 4K/128 is the median of three requests. The Qwen 32K/512 lane ran one request to completion. The two lanes used exactly 4,096 input and 128 output tokens, or 32,768 input and 512 output tokens. Image understanding was checked separately for multimodal profiles.
