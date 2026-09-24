# 4xT10 Profiles

## Qwen3.8-27B-FP8

Tested weight: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8); DFlash2 draft: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env` | 240K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
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

## Notes

1. Profiles are flat under each model/weight directory. Mode is selected by the launcher and defaults to `fast`.
2. Performance uses the launcher's reproducible reference lane: Prefix Cache is disabled only during testing, one text-only request is sent at a time, warm-up is excluded, 4K/128 is the median of three requests, and 32K/512 is run to completion. Multimodal profiles use the same text-only performance lane; image semantics are validated separately. `4K/128` means exactly 4,096 input tokens and 128 output tokens; `32K/512` means exactly 32,768 input tokens and 512 output tokens. Raw logs and request JSON remain in the external audit directory.
3. Test environment: the FP8/W8A16 profiles were validated on 2026-09-19 and the NVFP4/W4A16 profiles on 2026-09-23, both with v0.2.1. The host has two sockets with Intel Xeon E5-2673 v4 CPUs (80 logical CPUs total), 60 GiB RAM and 8 GiB swap. The tested topology uses physical GPUs 0, 2, 3 and 4, four Tesla T10 GPUs with 16,384 MiB each; all four are on the same NUMA node and connected by PCIe PIX links, and the service uses TP4 over this group. The host driver is NVIDIA 595.91.07. The runtime is the repository's `vllm-sm75-tp2-cu130` environment (CUDA 13.0, torch 2.13.0+cu130).
4. The `4x220K` route uses `GPU_UTIL=0.93` to reserve runtime headroom. Four simultaneous 32K requests passed; four simultaneous requests near the 220K limit are not a supported worst-case guarantee and can OOM on DFlash temporary buffers. Multi-concurrency profiles should not receive simultaneous ultra-long requests.
5. All profiles use the launcher's default `MAX_BATCHED_TOKENS=2048`; enable Prefix Cache in production for FP8KV routes. The measurements above deliberately disabled Prefix Cache. The W8A16 MTP4 FP8KV 2x262K route was additionally validated with two simultaneous 32K/512 requests at `GPU_UTIL=0.94`; both completed with exact token counts and the service remained healthy.
