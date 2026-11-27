# 2xT10 Profiles

Language: English | [简体中文](README.zh-CN.md)

## Qwen3.8-27B-INT4

Checkpoint [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env`](qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env) | 262,144 | FP8 | MTP/3 | text-only | 293,422 | 1176.34 / 87.28 | 1107.18 / 81.90 |
| [`qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env`](qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env) | 2 × 155,648 | TQK8V4 | MTP/3 | text-only | 351,112 | 1059.92 / 95.09 | 1069.71 / 107.12 |
| [`qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env`](qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env) | 2 × 220,000 | TQ4NC | MTP/3 | text-only | 511,566 | 1164.18 / 85.29 | 1122.17 / 73.27 |
| [`qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env`](qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env) | 196,000 | TQK8V4 | MTP/3 | text+image | 253,866 | 1191.44 / 82.76 | 1124.64 / 60.87 |

## Inkling 2 Mini

Official checkpoint [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

DSpark/7 loads `mtp.safetensors` from the official checkpoint and uses probabilistic draft sampling.

| Profile | Context | KV | Speculative decoding | Messages | GPU KV tokens |
| --- | ---: | --- | --- | --- | ---: |
| [`inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | AR | text-only | 12,602 |
| [`inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | DSpark/7 | text-only | 10,833 |

| Prompt | AR 4K/128 prefill / decode | DSpark/7 4K/128 prefill / decode | Draft acceptance |
| --- | ---: | ---: | ---: |
| Prose | 3644.31 / 63.67 | 3556.14 / 175.24 | 76.4–81.2% |
| Code | 3615.24 / 63.40 | 3540.27 / 183.17 | 86.5% |
| Repetitive | 3673.90 / 62.98 | 3594.91 / 278.34 | 100% |

## Capacity and feature checks

The FP8 KV capacity request completed 262,143 input tokens
and one output token in 476.5 seconds. The TQK8V4 capacity request completed
155,647 input tokens and one output token in 205.4 seconds. With prefix caching
enabled on the FP8 KV profile, the launcher
passed its smoke test, reported 288,504 GPU KV tokens, and a repeated request
hit 3,200 of 4,096 prompt tokens with identical output token IDs. The
text-image profile uses PIECEWISE CUDA graphs by default on Docker. With
MTP/3 enabled, it identified the blue square and orange circle in the
image smoke test.

## Test version and environment

- 2026-11-27, vLLM 2080 Ti Definitive Edition `v0.3.0.dev0`. The image uses official vLLM `v0.34.0`, CUDA 13.0, PyTorch `2.14.0+cu130`, and NVIDIA driver 610.43.03.
- 2 Tesla T10 GPUs with 16 GiB each, TP2, and PCIe P2P within one NUMA node. Tests used `fast` mode and a 2,048-token batch limit. Inkling used NCCL PCIe P2P.
- Throughput is in tok/s. After warmup, the tests sent one text-only request at a time with Prefix Cache disabled. 4K/128 is the median of three requests. The Qwen 32K/512 lane ran one request to completion. The two lanes used exactly 4,096 input and 128 output tokens, or 32,768 input and 512 output tokens. Image understanding was checked separately for multimodal profiles.
