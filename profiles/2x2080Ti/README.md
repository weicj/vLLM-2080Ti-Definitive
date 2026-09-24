# 2x2080Ti Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env` | 148K | FP16 | MTP/4 | text-only | 149,964 | 1643.37 / 97.55 | 1425.34 / 95.10 |
| `qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env` | 176K | FP16 | None (autoregressive) | text-only | 179,940 | 1694.94 / 33.17 | 1483.04 / 30.80 |
| `qwen27b/w8a16/nomtp-fp16kv-1x121K-text-image.env` | 121K | FP16 | None (autoregressive) | text+image | 122,608 | 1697.78 / 32.81 | 1447.04 / 30.65 |
| `qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env` | 262K | FP8 | MTP/4 | text-only | 295,455 | 1623.68 / 94.60 | 1393.94 / 92.47 |
| `qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env` | 186K | FP8 | MTP/4 | text+image | 190,540 | 1634.73 / 96.11 | 1384.95 / 93.17 |
| `qwen27b/w8a16/yarn-fp8kv-1x338K-text-only.env` | 338K | FP8 | None + YaRN | text-only | 354,143 | 1239.30 / 23.71 | 1338.48 / 17.91 |

## Qwen3.8-27B-NVFP4

Tested weight: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4); DFlash draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env` | 262K | FP8 | DFlash2/7 | text+image | 285,081 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| `qwen27b/w4a16/mtp4-fp8kv-2x229K-text-only.env` | 2 x 229K | FP8 | MTP/4 | text-only | 468,978 | 1432.48 / 116.04 | 1247.12 / 109.72 |
| `qwen27b/w4a16/yarn-fp8kv-1x524K-text-only.env` | 524K | FP8 | None + YaRN | text-only | 588,863 | 1466.29 / 41.36 | 1274.26 / 37.22 |
| `qwen27b/w4a16/dflash-fp8kv-2x176K-text-only.env` | 2 x 176K | FP8 | DFlash/7 | text-only | 357,194 | 1412.94 / 223.08 | 1249.27 / 211.10 |
| `qwen27b/w4a16/mtp4-tq4nc-3x262K-text-only.env` | 3 x 262K | TQ4NC | MTP/4 | text-only | 837,832 | 1446.95 / 127.86 | 1265.68 / 68.64 |

## Qwen3.6-35B-A3B-FP8

Tested weight: [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env` | 262K | FP16 | None (autoregressive) | text-only | 284,760 | 6690.07 / 113.63 | 5941.16 / 105.92 |
| `qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env` | 221K | FP8 | None (autoregressive) | text+image | 223,158 | 6357.56 / 110.25 | 5493.10 / 102.23 |

## Notes

1. Profiles are flat under each model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens. Raw logs and request JSON remain in the external audit directory.
3. NVFP4 W4A16 FP16KV has a significant quality-collapse issue and is not a recommended KV type.
4. The NVFP4 DFlash2 image route was validated with an explicit natural-language image question and returned the expected blue-square/orange-circle answer. The DFlash draft receives text-only inputs because it does not support external multimodal embeddings; the target model still performs the image understanding.

5. Test environment: validated on 2026-09-19 with v0.2.1. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The tested topology uses physical GPUs 1 and 5, both NVIDIA GeForce RTX 2080 Ti 22,528 MiB, connected by NV2; the service uses TP2 over this pair. The host driver is NVIDIA 595.91.07. The runtime is the repository's `vllm-sm75-tp2-cu130` environment (CUDA 13.0, torch 2.13.0+cu130).
