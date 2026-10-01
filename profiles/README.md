# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. They do not select the
checkpoint, GPUs, port, chat template, or reasoning defaults; those remain
launcher settings. Select the target with `MODEL_DIR` and the optional DFlash
draft with `SPECULATIVE_MODEL`.

Profiles are grouped by hardware, model family, and weight format. The shipped
layout is flat below each hardware/model/weight directory. The launcher
selects the startup mode, defaulting to `fast`.

```text
profiles/
  2x2080Ti/   # [profile details and reference performance](2x2080Ti/README.md)
  4xT10/      # [profile details and reference performance](4xT10/README.md)
  2xT10/      # [profile details and reference performance](2xT10/README.md)
```

Profile filenames use `<decoder>-<kv>-<concurrency><context>-<message>.env`.
The context label is the decimal-token value (floor of `MAX_MODEL_LEN / 1000`),
not a binary Ki-token conversion; for example, `262144` is labeled `262K`.
Thus `dflash2-tqk8v4-4x220K-text-only.env` is a DFlash2 route using TQK8V4 KV,
four concurrent requests, 220K decimal-token context per request, and text-only
messages. Decode routing uses `SPECULATIVE_METHOD=none|mtp|dflash` and
`SPECULATIVE_TOKENS`; the defaults are `0`, `3`, and `7`, respectively.
Per-request speculative metrics are a launcher setting:
`PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`.

To write a profile by hand, use one `KEY=value` assignment per line. The
required route fields are `MODEL_FAMILY`, `MODEL_VARIANT`, `QUANTIZATION`,
`KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, `GPU_UTIL`, `MAX_NUM_SEQS`, `MESSAGE_TYPE`,
`SPECULATIVE_METHOD`, and `SPECULATIVE_TOKENS`. Optional route fields are
`MODE`, `MAX_BATCHED_TOKENS`, and `ENABLE_YARN`.

For example, save the following as
`qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env`:

```dotenv
MODEL_FAMILY=qwen35
MODEL_VARIANT=fp8
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8
MAX_MODEL_LEN=262144
GPU_UTIL=0.96
MAX_NUM_SEQS=1
SPECULATIVE_METHOD=mtp
SPECULATIVE_TOKENS=4
MESSAGE_TYPE=text-only
```

| Profile | Mode | Context | KV | MTP | Messages | GPU KV tokens | Performance |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/fp16kv-104K-mtp3-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `qwen27b/w8a16/normal/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `qwen27b/w8a16/normal/fp8kv-220K-mtp3-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `qwen27b/w8a16/normal/fp8kv-256K-mtp3-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |
| `qwen27b/w8a16/normal/fp16kv-32K-mtp4-tp3-text-only.env` | normal | 32K | FP16 | 4 | text-only | 54,346 | 4K/128: 1025.5/73.0, 1290.3/74.8, 1204.9/75.0 tok/s; E2E 5.73/4.87/5.09s |
| `qwen27b/w8a16/normal/fp8kv-96K-mtp4-tp3-text-only.env` | normal | 96K | FP8 | 4 | text-only | 108,693 | warmup + 3x 4K/128: 1175.7/40.3, 1074.6/52.9, 1157.9/73.6 tok/s; E2E 6.64/6.21/5.26s |

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

Use `./launcher.sh --print-config` after selecting a profile to inspect the
resolved route before starting the service.

### TP3 routes

The `*-tp3-*` entries are route profiles for three T10/SM75 GPUs. Select the GPU
order and target TP outside the profile, for example
`GPU_DEVICES=6,7,8 TP_SIZE=3`. The measurements above use the Qwen3.8-27B-FP8
checkpoint, pure text, MTP4, prefix caching disabled, and the standard sequential
4K-input/128-output request window. DFlash2 TP3 is not promoted because its
independent draft parallelism is not validated on this route.
