# 4xT10 Profiles

语言：[English](README.md) | 简体中文

## Qwen3.8-27B-FP8

权重 [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

DFlash2 草稿权重 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env`](qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 331,998 | 1583.68 / 211.09 | 1466.17 / 210.14 |
| [`qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env`](qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env) | 240,384 | FP16 | DFlash2/7 | text+image | 293,387 | 1584.73 / 209.94 | 1443.93 / 218.32 |
| [`qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env`](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 1585.14 / 211.10 | 1344.57 / 209.08 |
| [`qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env`](qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 345,301 | 1578.10 / 121.59 | 1454.63 / 120.15 |
| [`qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env`](qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 650,056 | 1567.30 / 124.74 | 1428.35 / 125.80 |

## Qwen3.8-27B-NVFP4

权重 [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

DFlash2 草稿权重 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env`](qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 443,917 | 1559.39 / 249.82 | 1443.68 / 246.30 |
| [`qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env`](qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env) | 2 × 262,144 | FP8 | DFlash2/7 | text-only | 703,905 | 1561.66 / 247.63 | 1428.82 / 243.43 |
| [`qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env`](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| [`qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env`](qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 451,426 | 1557.56 / 143.11 | 1437.54 / 137.96 |
| [`qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env`](qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 853,104 | 1547.72 / 141.89 | 1412.98 / 124.51 |

## Inkling 2 Mini

官方权重 [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

此路线使用 Inkling 镜像。DSpark/7 从官方权重目录自动加载 `mtp.safetensors`，草稿采用概率采样。

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens |
| --- | ---: | --- | --- | --- | ---: |
| [`inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | 纯 AR | text-only | 24,503 |
| [`inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | DSpark/7 | text-only | 23,106 |

| 提示词 | 纯 AR 4K/128 prefill / decode | DSpark/7 4K/128 prefill / decode | 草稿接受率 |
| --- | ---: | ---: | ---: |
| 自然语言 | 4314.14 / 77.71 | 4172.03 / 255.02 | 81.2–86.5% |
| 代码 | 4312.15 / 77.37 | 4184.09 / 251.59 | 86.5% |
| 重复文本 | 4192.41 / 76.68 | 4089.48 / 353.55 | 78.6–100% |

## 容量与功能验证

Qwen 吞吐测试使用固定 KV 分配、LBNHC 布局（4×220K 除外）、FlashInfer PCIe IPC
all-reduce 和 launcher 的 `fast` 模式。测试期间关闭 Prefix Cache，240K 路线采用
0.96 的 GPU 显存利用率，其余 LBNHC 路线采用 0.94。240K 和 2×220K DFlash2
路线仅为解码捕获 CUDA Graph。五条图文路线均在图卡测试中识别出蓝色方块和橙色圆形。

DFlash2 使用 LBNHC 页布局时，镜像按 Qwen 目标模型的注意力层和循环层数分组，
避免按草稿层数填充目标层。

开启 Prefix Cache 时，launcher 为 240K 路线使用 `GPU_UTIL=0.965`，为 2×220K
路线使用 `GPU_UTIL=0.948` 和基于 PCIe P2P 的 NCCL。实测分别报告 295,117 和
508,888 个 GPU KV tokens。重复请求命中 4,096 个输入 token 中的 3,328 个，两次
生成 token ID 一致。2×220K 路线在开启缓存时还跑完两路并发 32K/512。

4×220K 路线使用 BLHNC 页布局与 `GPU_UTIL=0.93`。在 SM75 上，
FlashInfer 会在 DFlash 验证 token 之间复用解量化后的前缀。
四路同时发送的 32K/512 请求均完整生成，输入与输出 token 数符合测试口径。

NVFP4 权重的部分层采用 FP8 权重。镜像在 Turing 上对这些层和 NVFP4 投影均使用
Marlin。FP16 KV 路径使用单位缩放因子。

2×262K NVFP4 DFlash2 FP8 KV 路线完成了双路并发 32K/512。多路请求同时逼近配置的上下文上限时，DFlash 临时缓冲区仍可能耗尽显存。

## 测试版本与环境

- 2026-11-27，vLLM 2080 Ti Definitive Edition `v0.3.0.dev0`。镜像基于官方 vLLM `v0.34.0`，使用 CUDA 13.0、PyTorch `2.14.0+cu130` 和 NVIDIA 驱动 610.43.03。
- 4 张 16 GiB Tesla T10，TP4，同一 NUMA 节点内 PCIe P2P。测试采用 `fast` 模式，批处理上限为 2,048 token。Inkling 使用 NCCL PCIe P2P。
- 吞吐单位为 tok/s。预热后关闭 Prefix Cache，逐个发送纯文本请求。4K/128 取三次中位数，Qwen 的 32K/512 完整跑一次。4K/128 和 32K/512 分别是精确的 4,096 输入 / 128 输出和 32,768 输入 / 512 输出。图文 profile 的图像理解另行验证。
