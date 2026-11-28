# 2x2080Ti Profiles

Language: English | [简体中文](README.zh-CN.md)

The Qwen profiles carry over from 0.2.x and await 0.3.x Docker validation. See the Docker results for [2xT10](../2xT10/README.md) and [4xT10](../4xT10/README.md).

## Qwen3.8-27B-FP8

[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [mtp4-fp16kv-1x148K-text-only.env](qwen38-27b/w8a16/mtp4-fp16kv-1x148K-text-only.env) | 148K | FP16 | MTP/4 | text-only | 149,964 | 1643.37 / 97.55 | 1425.34 / 95.10 |
| [nomtp-fp16kv-1x176K-text-only.env](qwen38-27b/w8a16/nomtp-fp16kv-1x176K-text-only.env) | 176K | FP16 | None (autoregressive) | text-only | 179,940 | 1694.94 / 33.17 | 1483.04 / 30.80 |
| [nomtp-fp16kv-1x121K-text-image.env](qwen38-27b/w8a16/nomtp-fp16kv-1x121K-text-image.env) | 121K | FP16 | None (autoregressive) | text+image | 122,608 | 1697.78 / 32.81 | 1447.04 / 30.65 |
| [mtp4-fp8kv-1x262K-text-only.env](qwen38-27b/w8a16/mtp4-fp8kv-1x262K-text-only.env) | 262K | FP8 | MTP/4 | text-only | 295,455 | 1623.68 / 94.60 | 1393.94 / 92.47 |
| [mtp4-fp8kv-1x186K-text-image.env](qwen38-27b/w8a16/mtp4-fp8kv-1x186K-text-image.env) | 186K | FP8 | MTP/4 | text+image | 190,540 | 1634.73 / 96.11 | 1384.95 / 93.17 |
| [yarn-fp8kv-1x338K-text-only.env](qwen38-27b/w8a16/yarn-fp8kv-1x338K-text-only.env) | 338K | FP8 | None + YaRN | text-only | 354,143 | 1239.30 / 23.71 | 1338.48 / 17.91 |

## Qwen3.8-27B-NVFP4

[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [dflash2-fp8kv-1x262K-text-image.env](qwen38-27b/w4a16/dflash2-fp8kv-1x262K-text-image.env) | 262K | FP8 | DFlash2/7 | text+image | 285,081 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| [mtp4-fp8kv-2x229K-text-only.env](qwen38-27b/w4a16/mtp4-fp8kv-2x229K-text-only.env) | 2 x 229K | FP8 | MTP/4 | text-only | 468,978 | 1432.48 / 116.04 | 1247.12 / 109.72 |
| [yarn-fp8kv-1x524K-text-only.env](qwen38-27b/w4a16/yarn-fp8kv-1x524K-text-only.env) | 524K | FP8 | None + YaRN | text-only | 588,863 | 1466.29 / 41.36 | 1274.26 / 37.22 |
| [dflash-fp8kv-2x176K-text-only.env](qwen38-27b/w4a16/dflash-fp8kv-2x176K-text-only.env) | 2 x 176K | FP8 | DFlash/7 | text-only | 357,194 | 1412.94 / 223.08 | 1249.27 / 211.10 |
| [mtp4-tq4nc-3x262K-text-only.env](qwen38-27b/w4a16/mtp4-tq4nc-3x262K-text-only.env) | 3 x 262K | TQ4NC | MTP/4 | text-only | 837,832 | 1446.95 / 127.86 | 1265.68 / 68.64 |

## Qwen3.6-35B-A3B-FP8

[Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [nomtp-fp16kv-1x262K-text-only.env](qwen36-35b/w8a16/nomtp-fp16kv-1x262K-text-only.env) | 262K | FP16 | None (autoregressive) | text-only | 284,760 | 6690.07 / 113.63 | 5941.16 / 105.92 |
| [nomtp-fp8kv-1x221K-text-image.env](qwen36-35b/w8a16/nomtp-fp8kv-1x221K-text-image.env) | 221K | FP8 | None (autoregressive) | text+image | 223,158 | 6357.56 / 110.25 | 5493.10 / 102.23 |

## Test Notes

- The tables above report tests from September 19, 2026 on the 0.2.x native runtime (`v0.2.1`). The test pair used 2 × RTX 2080 Ti 22 GB with NVLink (NV2) and TP2, NVIDIA driver 595.91.07, CUDA 13.0, and PyTorch `2.13.0+cu130`. [Published profile guide](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/v0.2.2-post3/profiles/2x2080Ti/README.md)
- Throughput is in tok/s. The launcher sent one text-only request at a time with Prefix Cache disabled and excluded warm-up. 4K/128 reports the median of three requests, and 32K/512 reports one completed request. Input/output lengths were 4,096/128 and 32,768/512 tokens. Image understanding was checked separately.

## Inkling 2 Mini

Text and image profiles for 2 × RTX 2080 Ti 22 GiB in TP2. All use FP8 target KV, LBNHC and `GPU_UTIL=0.96`.

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

| Profile | Decoder | Concurrency × context per request | Startup KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | --- | ---: | ---: | ---: | ---: |
| [nomtp-fp8kv-1x1048K-text-image.env](inkling-2-mini/nomtp-fp8kv-1x1048K-text-image.env) | Pure AR | 1 × 1,048,576 | 1,230,629 | 2929.65 / 60.48 | 2688.90 / 57.70 |
| [nomtp-fp8kv-2x530K-text-image.env](inkling-2-mini/nomtp-fp8kv-2x530K-text-image.env) | Pure AR | 2 × 530,000 | 1,123,260 | 2584.57 / 104.70 | 2528.88 / 96.55 |
| [nomtp-fp8kv-3x320K-text-image.env](inkling-2-mini/nomtp-fp8kv-3x320K-text-image.env) | Pure AR | 3 × 320,000 | 1,006,906 | 2885.31 / 147.18 | 2588.91 / 127.83 |
| [nomtp-fp8kv-4x210K-text-image.env](inkling-2-mini/nomtp-fp8kv-4x210K-text-image.env) | Pure AR | 4 × 210,000 | 888,054 | 2952.78 / 186.48 | 2617.41 / 155.75 |
| [dflash2-fp8kv-1x870K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x870K-text-image.env) | DFlash2/7 | 1 × 870,000 | 926,614 | 2793.98 / 275.03 | 2580.39 / 207.40 |
| [dflash2-fp8kv-2x410K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x410K-text-image.env) | DFlash2/7 | 2 × 410,000 | 866,951 | 2626.30 / 408.58 | 2379.89 / 300.10 |
| [dflash2-fp8kv-3x220K-text-image.env](inkling-2-mini/dflash2-fp8kv-3x220K-text-image.env) | DFlash2/7 | 3 × 220,000 | 677,026 | 2601.92 / 493.51 | 2376.21 / 298.19 |
| [dflash2-fp8kv-4x140K-text-image.env](inkling-2-mini/dflash2-fp8kv-4x140K-text-image.env) | DFlash2/7 | 4 × 140,000 | 573,531 | 2736.95 / 594.34 | 2427.26 / 369.16 |
| [dspark7-fp8kv-1x900K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x900K-text-image.env) | DSpark/7 | 1 × 900,000 | 930,394 | 2700.40 / 233.86 | 2468.55 / 185.43 |
| [dspark7-fp8kv-2x430K-text-image.env](inkling-2-mini/dspark7-fp8kv-2x430K-text-image.env) | DSpark/7 | 2 × 430,000 | 897,728 | 2575.34 / 354.87 | 2431.29 / 263.60 |
| [dspark7-fp8kv-3x270K-text-image.env](inkling-2-mini/dspark7-fp8kv-3x270K-text-image.env) | DSpark/7 | 3 × 270,000 | 848,950 | 1379.51 / 414.08 | 1331.07 / 282.97 |
| [dspark7-fp8kv-4x190K-text-image.env](inkling-2-mini/dspark7-fp8kv-4x190K-text-image.env) | DSpark/7 | 4 × 190,000 | 795,839 | 1706.33 / 491.95 | 1545.74 / 336.36 |

- Tested with vLLM 2080 Ti Definitive Edition `v0.3.0`, upstream vLLM `v0.34.0`, PyTorch `2.14.0+cu130`, CUDA 13.0 and NVIDIA driver 595.71.05. The validation host used PCIe with NCCL P2P disabled. The patched runtime was extracted from the image and run as a native process with the launcher's `fast` arguments.
- Each request continued `1 2 3 4` with one 480 × 480 image and prefix caching disabled. Both workloads run at the listed concurrency and report medians over three rounds in tok/s. Concurrent prefill ends at the latest first token. Concurrent decode sums tokens across requests in the shared decode window, trimming 10% from each end up to 250 ms per end. No requests were preempted. Startup logs report KV capacity, and throughput tests use up to 32,768 input tokens per request.
- DFlash2 uses FP16 draft KV and reported 100% per-request draft acceptance on these benchmark requests. DSpark loads `mtp.safetensors` from the main checkpoint.
