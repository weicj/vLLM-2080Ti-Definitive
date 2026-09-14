# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. They do not select the
checkpoint, GPUs, port, chat template, or reasoning defaults; those remain
launcher settings. Select the model path separately with `MODEL_DIR`.

The shipped layout is `model / weight / mode`:

```text
profiles/
  qwen27b/
    w8a16/                 # Qwen3.8 FP8 weights
      normal/
      fast/
    w4a16/                 # Qwen3.8 NVFP4 weights
      normal/
      fast/
  qwen35b/
    w8a16/                 # Qwen3.x 35B FP8 weights
      normal/
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

### [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/fp16kv-104K-mtp3-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `qwen27b/w8a16/normal/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `qwen27b/w8a16/normal/fp8kv-220K-mtp3-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `qwen27b/w8a16/normal/fp8kv-256K-mtp3-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |

### [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env` | normal | 240K | FP8 | 3 | text-only | 463,890 | 1433.2 / 76.8 |
| `qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-image.env` | normal | 240K | FP8 | 3 | text+image | 426,080 | 1250.6 / 52.5 |
| `qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 | 0 | text-only | 518,191 | 1372.1 / 42.0 |
| `qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC | 3 | text-only | 732,381 | 1402.9 / 103.5 |

### Concurrent benchmark lanes (NVFP4 text-only)

| Profile | Mode | Context | KV/MTP | GPU KV tokens | C1 | C2 | C4 | C8 | Evidence |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 / 0 | 518,191 | 1372.1 / 42.0 | 1507.4 / 80.4 | 1535.9 / 152.0 | 1523.4 / 270.8 | full-window run |
| `qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC / 3 | 732,381 | 1402.9 / 103.5 | 1449.1 / 180.4 | 1460.0 / 220.7 | 1449.1 / 347.3 | full-window run |

Each C cell is `prefill / full-window aggregate decode` tok/s; prefix caching
was disabled.

### [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 273,586 | 7378 / 128.7 |
| `qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 146,485 | 5965.8 / 127.6 |

### Qwen3.8 Flash-Next EXL3 TP2xPP2

`qwen38flashnext/exl3/experimental/tp2pp2-ssd-nomtp-text.env` is the initial
four-GPU T10 functional-validation route for the native ExLlamaV3 EXL3 pack.
It uses TP=2, PP=2, and expert parallelism so each local expert retains its
640-wide intermediate dimension, and enables SSD-backed PLE n-gram streaming
through the launcher. Install the pinned
`vllm-exl3-turing`/`exllamav3-turing` pair described in
[`docs/usage/exl3_turing.md`](../docs/usage/exl3_turing.md) before starting it.
No throughput number is recorded until a real TP2xPP2 CUDA-Graph run passes
output and loader-path checks.
Use `./launcher.sh --print-config` after selecting a profile to inspect the
resolved route before starting the service.
