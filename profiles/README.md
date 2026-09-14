# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. Hardware-specific profiles
may also select `TP_SIZE`; checkpoint, GPUs, port, chat template, and reasoning
defaults remain launcher settings. Select the model path separately with
`MODEL_DIR`.

The shipped layout is `hardware / model / weight / mode`:

```text
profiles/
  2x2080Ti/
    qwen27b/
      w8a16/               # Qwen3.8 FP8 weights
      w4a16/               # Qwen3.8 NVFP4 weights
    qwen35b/
      w8a16/               # Qwen3.x 35B FP8 weights
  4xT10/
    qwen27b/
      w8a16/               # Qwen3.8 FP8 weights, TP=4 with CAR
```

`w8a16` means FP8 weights with FP16 activations; `w4a16` means NVFP4 weights
with FP16 activations. `normal`, `fast`, and `aggressive` are the startup
modes encoded by `COMPATIBLE_MODES`. `MTP_K=0` means no MTP; `MTP_K=3` means
MTP3. The message type is explicit in every shipped profile as
`MESSAGE_TYPE=text-only` or `MESSAGE_TYPE=text+image`.

## Profile and Measured Performance

<small>Historical pre3 reference environment: `.31`, dual Intel Xeon E5-2673 v4 (40 cores / 80
threads), 60 GiB RAM + 8 GiB swap, dual RTX 2080 Ti 22 GiB with NVLink (SM75),
Ubuntu 26.04, kernel 7.0.0-30, driver 595.84, CUDA 13.0, PyTorch 2.13, vLLM
0.27.1, TP=2/PP=1, CUDA Graph, no eager execution. Performance is 4K input /
128 output, prefill / decode tok/s. These figures do not validate pre4, which
uses upstream nightly `b23433088b`.</small> `-` means no stable measurement. See
the [historical validation record](../docs/2080ti-0.2.1-pre-validation.md) for details.

The 4xT10 rows were measured on four 16 GiB Tesla T10 GPUs over PCIe (TP=4),
using pre4 `315d5930f5` plus the ABI-matched PCIe CAR extension from
[#153](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/153)
(`a910fe9e43`). They require that extension; unpatched pre4 is not an
equivalent route. Both runs retained custom all-reduce and logged
`['CUSTOM', 'PYNCCL']`, passed startup and an image request, and report the
median of three independent 4K/128 requests with distinct prompts. The fast
TQK8V4 route used FlashInfer FA2 on SM75 and completed full and piecewise CUDA
Graph capture. Fixed 32-question GSM8K smoke scores were 21/32 for normal
FP16 and 26/32 for fast TQK8V4; 28,828-token retrieval, multi-turn, image OCR,
and WAL semantic probes also passed. The image smoke validates visual request
routing and generation, not visual-answer accuracy.

### [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-104K-mtp3-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `2x2080Ti/qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `2x2080Ti/qwen27b/w8a16/normal/fp8kv-220K-mtp3-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `2x2080Ti/qwen27b/w8a16/normal/fp8kv-256K-mtp3-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |
| `4xT10/qwen27b/w8a16/normal/fp16kv-256K-mtp3-text-image.env` | normal | 256K | FP16 | 3 | text+image | 312,585 | 1440.31 / 73.68 |
| `4xT10/qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-image.env` | fast | 256K | TQK8V4 | 3 | text+image | 780,814 | 1692.12 / 104.78 |

### [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env` | normal | 240K | FP8 | 3 | text-only | 463,890 | 1433.2 / 76.8 |
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-image.env` | normal | 240K | FP8 | 3 | text+image | 426,080 | 1250.6 / 52.5 |
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 | 0 | text-only | 518,191 | 1372.1 / 42.0 |
| `2x2080Ti/qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC | 3 | text-only | 732,381 | 1402.9 / 103.5 |

### Concurrent benchmark lanes (NVFP4 text-only)

| Profile | Mode | Context | KV/MTP | GPU KV tokens | C1 | C2 | C4 | C8 | Evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 / 0 | 518,191 | 1372.1 / 42.0 | 1507.4 / 80.4 | 1535.9 / 152.0 | 1523.4 / 270.8 | full-window run |
| `2x2080Ti/qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC / 3 | 732,381 | 1402.9 / 103.5 | 1449.1 / 180.4 | 1460.0 / 220.7 | 1449.1 / 347.3 | full-window run |

Each C cell is `prefill / full-window aggregate decode` tok/s; prefix caching
was disabled.

### [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 273,586 | 7378 / 128.7 |
| `2x2080Ti/qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 146,485 | 5965.8 / 127.6 |

Use `./launcher.sh --print-config` after selecting a profile to inspect the
resolved route before starting the service.
