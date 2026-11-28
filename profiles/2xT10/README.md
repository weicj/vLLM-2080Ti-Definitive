# 2xT10 Profiles

Language: English | [简体中文](README.zh-CN.md)

## Qwen3.8-27B-INT4

Checkpoint [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [mtp3-fp8kv-1x262K-text-only.env](qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env) | 262,144 | FP8 | MTP/3 | text-only | 293,422 | 1176.34 / 87.28 | 1107.18 / 81.90 |
| [mtp3-tqk8v4-2x155K-text-only.env](qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env) | 2 × 155,648 | TQK8V4 | MTP/3 | text-only | 351,112 | 1059.92 / 95.09 | 1069.71 / 107.12 |
| [mtp3-tq4nc-2x220K-text-only.env](qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env) | 2 × 220,000 | TQ4NC | MTP/3 | text-only | 511,566 | 1164.18 / 85.29 | 1122.17 / 73.27 |
| [mtp3-tqk8v4-1x196K-text-image.env](qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env) | 196,000 | TQK8V4 | MTP/3 | text+image | 253,866 | 1196.41 / 87.30 | 1136.75 / 60.36 |

## Inkling 2 Mini

Weights and draft [Official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16 draft](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

All profiles use FP8 target KV. DSpark/7 loads `mtp.safetensors` from the official checkpoint. DFlash2/7 uses FP16 draft KV and the LBNHC layout.

| Profile | Context | Test concurrency | Startup KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| [dspark7-fp8kv-2x200K-text-only.env](inkling-2-mini/dspark7-fp8kv-2x200K-text-only.env) | 2 × 200,000 | 2 | 411,276 | 2003.21 / 338.11 | 1911.51 / 202.07 |
| [dspark7-fp8kv-1x262K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x262K-text-image.env) | 1 × 262,144 | 1 | 306,544 | 2480.39 / 249.43 | 2419.47 / 165.72 |
| [dflash2-fp8kv-1x262K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x262K-text-image.env) | 1 × 262,144 | 1 | 317,129 | 2465.67 / 237.90 | 2418.85 / 174.58 |
| [dflash2-fp8kv-2x128K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x128K-text-image.env) | 2 × 128,000 | 2 | 267,636 | 2009.46 / 376.91 | 2113.50 / 236.45 |
| [dflash2-fp8kv-2x176K-text-only.env](inkling-2-mini/dflash2-fp8kv-2x176K-text-only.env) | 2 × 176,000 | 2 | 369,278 | 2370.57 / 384.04 | 2108.90 / 238.03 |
| [nomtp-fp8kv-1x522K-text-image.env](inkling-2-mini/nomtp-fp8kv-1x522K-text-image.env) | 1 × 522,000 | 1 | 523,533 | 3217.41 / 48.47 | 2858.22 / 46.10 |
| [nomtp-fp8kv-2x203K-text-image.env](inkling-2-mini/nomtp-fp8kv-2x203K-text-image.env) | 2 × 203,000 | 2 | 409,366 | 1923.62 / 87.11 | 2257.81 / 78.92 |

Dual-GPU DSpark/7 text-image launch example

```bash
MODEL_DIR=/path/to/Inkling-2-Mini \
PROFILE=2xT10/inkling-2-mini/dspark7-fp8kv-1x262K-text-image.env \
GPU_DEVICES=0,1 TP_SIZE=2 NCCL_P2P_DISABLE=0 DISABLE_PREFIX_CACHING=1 \
NON_INTERACTIVE=1 ./launcher.sh
```

## Test Notes

- **Environment** for tests run November 27–28, 2026. vLLM 2080 Ti Definitive Edition `v0.3.0` on the official vLLM `v0.34.0` Docker base. CUDA 13.0, PyTorch `2.14.0+cu130`, NVIDIA driver 610.43.03, and 2 × 16 GiB Tesla T10 in TP2 with PCIe P2P. Tests used fast mode with Prefix Cache disabled. Inkling used NCCL, with vLLM custom all-reduce enabled for pure AR.
- **Throughput** is in tok/s. 4K/128 and 32K/512 use 4,096/128 and 32,768/512 input/output tokens per request. Concurrent prefill ends at the latest first token. Concurrent decode counts output only in the shared decode window, trimmed by 10% at each end up to 250 ms.
- **Qwen** uses one text request at a time and a 2,048-token batch limit. The 4K/128 result is a three-run median. The 32K/512 test ran once. The 196K profile used `1 2 3 4` with 100% draft acceptance. Other rows used the fixed token ` the`. The 262K FP8 KV and 155K TQK8V4 profiles completed full-context tests.
- **Inkling** DSpark/7 and DFlash2/7 runs use high-acceptance `1 2 3 4` continuation, with one 480 × 480 image for each multimodal request. 4K/128 uses three-round medians. At 32K/512, DSpark/7 uses two rounds and DFlash2/7 uses three. The DSpark/7 1 × 262K multimodal profile passed a separate full-context test. DFlash2/7 had 100% draft acceptance and zero preemptions.
- **Pure AR** uses LBNHC and 97% GPU memory utilization. Context limits are sized from startup KV capacity with a small margin. 4K/128 and 32K/512 are three-run medians at the listed concurrency with one image per request.
