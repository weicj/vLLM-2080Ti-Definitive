# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

This directory contains launch profiles for vLLM 2080Ti Definitive. A profile
is an `.env` preset for runtime parameters only; it does not include the
checkpoint path. Choose the model directory separately in `launcher.sh` or with
`MODEL_DIR=...`.

The shipped context and throughput numbers were validated on 2x RTX 2080 Ti
22GB cards with tensor parallel size 2.

Profile layout:

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
  qwen35b/
    normal/
      fp8/
    aggressive/
      fp8/
    fast/
      fp8/
    user/
  qwen38flashnext/
    w4a16/
      experimental/
```

Launch modes:

- `safe`: conservative fallback mode for maximum compatibility.
- `normal`: recommended daily mode for stable deployments.
- `fast`: high-performance mode for higher throughput.
- `aggressive`: more aggressive mode with the highest performance and quality risk.

`profiles/templates/` contains optional chat-template presets. They are selected
from the launcher as a global service setting; route profiles do not store chat
templates, GPU devices, ports, reasoning defaults, or tool-calling defaults.
`MODEL_FAMILY` records the checkpoint architecture rather than the marketing
model name: `qwen35` is Qwen3.5-compatible dense, `qwen35moe` is its MoE
variant, and `qwen4` is Qwen4-Exp/Flash-Next. The launcher reads `config.json`
(`model_type` and `architectures`) and uses this value to filter profiles.
For the shipped Qwen3/Qwen3.6 routes, the launcher fills in `qwen3` as the
reasoning parser when one is not set so startup smoke and chat parsing stay
aligned with the model's default reasoning behavior. Set `REASONING_PARSER=off`
if you need to run without a reasoning parser for diagnostics.

Qwen4 profiles may set `PLE_PLACEMENT=disk|cpu|gpu`. `disk` is the validated
default and reads PLE rows directly from safetensors mappings in the CPU worker;
`cpu` loads the complete PLE table into system RAM; `gpu` keeps it in VRAM and
requires enough memory for the full table. The shipped T10 profiles use `disk`.

File names describe the intended route:

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV positioning:

- `fp16kv`: quality route.
- `int8kv`: capacity / balance route; currently shipped only as `normal`
  profiles.
- `tqk8v4`: TurboQuant K8V4 compression route; currently shipped only for
  quality-passed `fast` profiles.
- Official Qwen3.6 35B currently ships as FP8 weight + FP16 KV presets for
  both text-only and text+image.

The shipped TQK8V4 profiles use `MAX_BATCHED_TOKENS=2560`, which is the
validated setting for the prefix-cache path with aligned Qwen hybrid cache
blocks.

## Validated Profiles

### Qwen3.6 27B FP8

Tested checkpoint: Jackrong/Qwopus3.6-27B-v2-FP8, about 29G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### Qwen3.6 35B FP8

Tested checkpoint target: Qwen/Qwen3.6-35B-A3B-FP8, about 36G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.8 27B NVFP4 DFlash2

Validated target checkpoint: `nvidia/Qwen3.8-27B-NVFP4`; draft checkpoint:
`incoai/Qwen3.8-27B-DFlash2`. This is the first promoted DFlash2 route for the
dual RTX 2080 Ti SM75 runtime. It uses TurboQuant K8V4 KV, K=7, causal target
verification in B=8 chunks, TRITON_ATTN for the draft, and PIECEWISE CUDA
Graphs. The measured lane is high-acceptance fixed 4K prompt / 128 completion;
values are prefill / decode tok/s.

| Profile | Compatible modes | Context | KV | DFlash K | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env` | normal | 256K | TQK8V4 | 7 | text-only | 1 | **~1450 / 162.83** |

Evidence: three independent 4K/128 runs measured 155.16, 166.57, and
166.76 decode tok/s (mean 162.83), all returned 128/128 tokens and completed
the stream. Three 4K correctness requests returned exactly `PROFILE_OK`, and
the 262,144-token service configuration exposed 331,935 KV tokens at startup.
The profile is validated for text-only requests; B=8 is an explicit SM75
tuning parameter and must retain the safe causal verification branch.

### Qwen3.8 Flash-Next NVFP4 experimental PP routes

These profiles target eight Tesla T10 GPUs on host `.31` (`0,2,3,4,6,7,8,9`)
with ModelOpt NVFP4 through the SM75 Marlin W4A16 fallback, disk-mapped PLE offload,
FP16 KV, no MTP, text-only serving, synchronous scheduling, non-eager CUDA Graphs, and
`MAX_BATCHED_TOKENS=512`. Measurements are fixed-token 4K prompt / 128
completion runs; values are median prefill / decode tok/s.

| Profile | TP/PP | Backend | Prefill / decode tok/s | Route guidance |
|---|---:|---|---:|---|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 2x4 | FlashQLA + PYNCCL | **2379.14 / 24.92** | Recommended balance; default `GPU_UTIL=0.92` |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 4x2 | FlashQLA + PYNCCL | **1697.99 / 28.18** | Decode-priority; current-util capacity profile |

The earlier `33.93 tok/s` TP4xPP2 observation is a shape-dependent peak, not a
stable service guarantee; repeated runs are normally about 27--28 tok/s. CAR
(custom all-reduce) `auto` mode opts into topology detection: CAR is used only
for TP groups that are fully NVLink-connected and P2P-reachable. PCIe-only T10
groups use NCCL/PYNCCL instead.

The same fixed profile settings exposed the following GPU KV-token capacities
at the shipped utilization defaults. These are capacity/startup measurements,
not additional throughput samples.

| Profile | Hardware reference | GPU util | GPU KV tokens |
|---|---|---:|---:|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 8x Tesla T10 | 0.92 | 102,591 |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 8x Tesla T10 | 0.96 | 100,031 |
| TP2xPP4 heterogeneous reference | 6x Tesla T10 + 2x RTX 2080 Ti; TP groups `[0,2]`, `[8,9]`, `[6,7]`, `[1,5]` | 0.92 | **152,492** |

These are current-environment measurements. The heterogeneous row is a
capacity/startup reference from a pre2-labelled launcher log (health and real
requests succeeded); re-run it on the current pre3 tree before treating it as
a pre3 capacity promotion. The homogeneous rows use the eight T10 UUIDs. The
TP2xPP4 default completed QSA JIT warmup followed by ten sequential 4K/128
requests; its mean was 2469.26 / 24.90 tok/s and the table reports the median.
TP2xPP4 at `GPU_UTIL=0.95` exposed 115,834 tokens but its ten-run median fell
to 1274.87 / 21.15 tok/s. At `0.97`, startup exposed 124,662 tokens but the
request failed with a FlashQLA temporary-buffer OOM. After three warmups, the
listed TP4xPP2 `0.96` route completed ten runs without OOM; its all-sample mean
was 1652.15 / 27.33 tok/s, while eight of ten decode samples clustered at a
28.33 tok/s mean (28.31 median).

### Qwen3.6 27B AWQ/GPTQ-INT4

Tested checkpoints: QuantTrio/Qwen3.6-27B-AWQ, mconcat/Qwopus3.6-27B-v2-AWQ-4bit,
and llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4,
about 19G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 250K per workspace | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 250K per workspace | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

## Recent INT8KV 4K Reference

These short-lane numbers supplement the large-context entries above. They were
revalidated on the current shipped `normal` INT8KV routes with the repository's
`PP4096/TG128` synthetic single-request lane.

| Profile | Benchmark lane | Throughput | Notes |
|---|---|---:|---|
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | PP4096/TG128 | 1557.20 / 73.79 | Current shipped FP8 INT8KV `normal` reference. |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | PP4096/TG128 | 1684.19 / 69.53 | Current shipped GPTQ-INT4 INT8KV `normal` reference. |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | PP4096/TG128 | 1679.50 / 41.83 | Current shipped GPTQ-INT4 INT8KV + YaRN `normal` reference. |

These short-lane rows supplement the validated large-context entries in the
main tables above; they do not replace the long-context capacity evidence.
