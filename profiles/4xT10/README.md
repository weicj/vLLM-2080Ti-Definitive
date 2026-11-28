# 4xT10 Profiles

Language: English | [简体中文](README.zh-CN.md)

## Qwen3.8-27B-FP8

Checkpoint [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · DFlash2 draft checkpoint [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | Test concurrency | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| [dflash2-fp16kv-1x262K-text-only.env](qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 331,998 | 1 | 1583.68 / 211.09 | 1466.17 / 210.14 |
| [dflash2-fp16kv-1x240K-text-image.env](qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env) | 240,384 | FP16 | DFlash2/7 | text+image | 293,387 | 1 | 1584.73 / 209.94 | 1443.93 / 218.32 |
| [dflash2-fp8kv-2x220K-text-only.env](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 1 | 1340.62 / 210.43 | 1427.11 / 208.97 |
| [dflash2-fp8kv-2x220K-text-only.env](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 2 | 1670.08 / 385.06 | 1487.07 / 355.14 |
| [mtp4-fp16kv-1x262K-text-image.env](qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 345,301 | 1 | 1578.10 / 121.59 | 1454.63 / 120.15 |
| [mtp4-fp8kv-2x262K-text-image.env](qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 650,056 | 1 | 1567.30 / 124.74 | 1428.35 / 125.80 |

## Qwen3.8-27B-NVFP4

Checkpoint [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · DFlash2 draft checkpoint [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | Test concurrency | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| [dflash2-fp16kv-1x262K-text-only.env](qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 443,917 | 1 | 1559.39 / 249.82 | 1443.68 / 246.30 |
| [dflash2-fp8kv-2x262K-text-only.env](qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env) | 2 × 262,144 | FP8 | DFlash2/7 | text-only | 703,905 | 1 | 1561.66 / 247.63 | 1428.82 / 243.43 |
| [dflash2-tqk8v4-4x220K-text-only.env](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 1 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| [dflash2-tqk8v4-4x220K-text-only.env](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 4 | 1585.96 / 797.18 | 1540.44 / 711.94 |
| [mtp4-fp16kv-1x262K-text-image.env](qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 451,426 | 1 | 1557.56 / 143.11 | 1437.54 / 137.96 |
| [mtp4-fp8kv-2x262K-text-image.env](qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 853,104 | 1 | 1547.72 / 141.89 | 1412.98 / 124.51 |

## Inkling 2 Mini

Weights and draft [Official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16 draft](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

All profiles use FP8 target KV. DSpark/7 loads `mtp.safetensors` from the official checkpoint. DFlash2/7 uses FP16 draft KV and the LBNHC layout.

| Profile | Context | Test concurrency | Startup KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| [dspark7-fp8kv-1x1048K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x1048K-text-image.env) | 1 × 1,048,576 | 1 | 1,252,517 | 3495.49 / 314.43 | 3517.83 / 232.72 |
| [dspark7-fp8kv-2x500K-text-image.env](inkling-2-mini/dspark7-fp8kv-2x500K-text-image.env) | 2 × 500,000 | 2 | 1,438,512 | 2014.68 / 405.64 | 2047.55 / 273.72 |
| [dspark7-fp8kv-4x250K-text-image.env](inkling-2-mini/dspark7-fp8kv-4x250K-text-image.env) | 4 × 250,000 | 4 | 1,256,622 | 1539.25 / 570.08 | 1288.98 / 372.64 |
| [dflash2-fp8kv-1x1048K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x1048K-text-image.env) | 1 × 1,048,576 | 1 | 1,258,521 | 3729.50 / 290.63 | 3480.09 / 232.84 |
| [dflash2-fp8kv-2x512K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x512K-text-image.env) | 2 × 512,000 | 2 | 1,083,356 | 3398.49 / 444.66 | 3353.87 / 338.78 |
| [dflash2-fp8kv-4x250K-text-image.env](inkling-2-mini/dflash2-fp8kv-4x250K-text-image.env) | 4 × 250,000 | 4 | 1,013,599 | 2966.63 / 653.72 | 2992.04 / 427.90 |

## Test Notes

- **Environment** for tests run November 27–28, 2026. vLLM 2080 Ti Definitive Edition `v0.3.0` on the official vLLM `v0.34.0` Docker base. CUDA 13.0, PyTorch `2.14.0+cu130`, NVIDIA driver 610.43.03, and 4 × 16 GiB Tesla T10 in TP4 with PCIe P2P. Tests used fast mode with Prefix Cache disabled. Inkling used NCCL.
- **Throughput** is in tok/s. 4K/128 and 32K/512 use 4,096/128 and 32,768/512 input/output tokens per request. Concurrent prefill ends at the latest first token. Concurrent decode counts output only in the shared decode window, trimmed by 10% at each end up to 250 ms.
- **Qwen** uses text requests and a 2,048-token batch limit. FP8 2 × 220K uses `1 2 3 4`. Other rows use the fixed token ` the`. 4K/128 is a three-round median. 32K/512 used one round at concurrency 1 and three rounds for concurrent tests. On one PCIe switch, the dual-request FP8 and four-request NVFP4 tests had 100% draft acceptance and zero preemptions. NVFP4 2 × 262K also completed two concurrent 32K/512 requests. DFlash temporary buffers can exhaust VRAM when multiple requests approach the context limit.
- **Inkling** DSpark/7 and DFlash2/7 runs use high-acceptance `1 2 3 4` continuation and one 480 × 480 image per request. 4K/128 uses three-round medians. At 32K/512, DSpark/7 uses two rounds and DFlash2/7 uses three. The four-request DFlash2 retest used one PCIe switch, a 1024-token batch limit, and a 256-token prefill threshold. Long-context throughput tests reach 32,768 input tokens per request. The table uses startup KV estimates for capacity. DFlash2/7 had 100% draft acceptance and zero preemptions.
