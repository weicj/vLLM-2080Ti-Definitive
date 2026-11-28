# 2x2080Ti Profiles

语言：[English](README.md) | 简体中文

Qwen 配置沿用 0.2.x，等待 0.3.x Docker 验证。Docker 实测见 [2xT10](../2xT10/README.zh-CN.md) 和 [4xT10](../4xT10/README.zh-CN.md)。

## Qwen3.8-27B-FP8

[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [mtp4-fp16kv-1x148K-text-only.env](qwen38-27b/w8a16/mtp4-fp16kv-1x148K-text-only.env) | 148K | FP16 | MTP/4 | text-only | 149,964 | 1643.37 / 97.55 | 1425.34 / 95.10 |
| [nomtp-fp16kv-1x176K-text-only.env](qwen38-27b/w8a16/nomtp-fp16kv-1x176K-text-only.env) | 176K | FP16 | 无/自回归 | text-only | 179,940 | 1694.94 / 33.17 | 1483.04 / 30.80 |
| [nomtp-fp16kv-1x121K-text-image.env](qwen38-27b/w8a16/nomtp-fp16kv-1x121K-text-image.env) | 121K | FP16 | 无/自回归 | text+image | 122,608 | 1697.78 / 32.81 | 1447.04 / 30.65 |
| [mtp4-fp8kv-1x262K-text-only.env](qwen38-27b/w8a16/mtp4-fp8kv-1x262K-text-only.env) | 262K | FP8 | MTP/4 | text-only | 295,455 | 1623.68 / 94.60 | 1393.94 / 92.47 |
| [mtp4-fp8kv-1x186K-text-image.env](qwen38-27b/w8a16/mtp4-fp8kv-1x186K-text-image.env) | 186K | FP8 | MTP/4 | text+image | 190,540 | 1634.73 / 96.11 | 1384.95 / 93.17 |
| [yarn-fp8kv-1x338K-text-only.env](qwen38-27b/w8a16/yarn-fp8kv-1x338K-text-only.env) | 338K | FP8 | 无 + YaRN | text-only | 354,143 | 1239.30 / 23.71 | 1338.48 / 17.91 |

## Qwen3.8-27B-NVFP4

[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [dflash2-fp8kv-1x262K-text-image.env](qwen38-27b/w4a16/dflash2-fp8kv-1x262K-text-image.env) | 262K | FP8 | DFlash2/7 | text+image | 285,081 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| [mtp4-fp8kv-2x229K-text-only.env](qwen38-27b/w4a16/mtp4-fp8kv-2x229K-text-only.env) | 2 x 229K | FP8 | MTP/4 | text-only | 468,978 | 1432.48 / 116.04 | 1247.12 / 109.72 |
| [yarn-fp8kv-1x524K-text-only.env](qwen38-27b/w4a16/yarn-fp8kv-1x524K-text-only.env) | 524K | FP8 | 无 + YaRN | text-only | 588,863 | 1466.29 / 41.36 | 1274.26 / 37.22 |
| [dflash-fp8kv-2x176K-text-only.env](qwen38-27b/w4a16/dflash-fp8kv-2x176K-text-only.env) | 2 x 176K | FP8 | DFlash/7 | text-only | 357,194 | 1412.94 / 223.08 | 1249.27 / 211.10 |
| [mtp4-tq4nc-3x262K-text-only.env](qwen38-27b/w4a16/mtp4-tq4nc-3x262K-text-only.env) | 3 x 262K | TQ4NC | MTP/4 | text-only | 837,832 | 1446.95 / 127.86 | 1265.68 / 68.64 |

## Qwen3.6-35B-A3B-FP8

[Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| [nomtp-fp16kv-1x262K-text-only.env](qwen36-35b/w8a16/nomtp-fp16kv-1x262K-text-only.env) | 262K | FP16 | 无/自回归 | text-only | 284,760 | 6690.07 / 113.63 | 5941.16 / 105.92 |
| [nomtp-fp8kv-1x221K-text-image.env](qwen36-35b/w8a16/nomtp-fp8kv-1x221K-text-image.env) | 221K | FP8 | 无/自回归 | text+image | 223,158 | 6357.56 / 110.25 | 5493.10 / 102.23 |

## 测试说明

- 上表数据均于 2026-09-19 在 0.2.x 原生运行时（`v0.2.1`）下测得。测试使用 2 × RTX 2080 Ti 22 GB，NVLink（NV2）、TP2、NVIDIA 驱动 595.91.07、CUDA 13.0 和 PyTorch `2.13.0+cu130`。[发布版 Profile 指南](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/v0.2.2-post3/profiles/2x2080Ti/README.zh-CN.md)
- 吞吐单位为 tok/s。Launcher 每次发送一条纯文本请求，关闭 Prefix Cache，预热不计入统计。4K/128 取三次中位数，32K/512 完整跑一次，输入和输出分别为 4,096/128 和 32,768/512 tokens。图像理解另行验证。

## Inkling 2 Mini

适用于 2 × 2080 Ti 22 GiB、TP2 的图文配置。统一使用 FP8 主模型 KV、LBNHC 和 `GPU_UTIL=0.96`。

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

| Profile | 解码方式 | 并发 × 每路上下文 | 启动 KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | --- | ---: | ---: | ---: | ---: |
| [nomtp-fp8kv-1x1048K-text-image.env](inkling-2-mini/nomtp-fp8kv-1x1048K-text-image.env) | 纯 AR | 1 × 1,048,576 | 1,230,629 | 2929.65 / 60.48 | 2688.90 / 57.70 |
| [nomtp-fp8kv-2x530K-text-image.env](inkling-2-mini/nomtp-fp8kv-2x530K-text-image.env) | 纯 AR | 2 × 530,000 | 1,123,260 | 2584.57 / 104.70 | 2528.88 / 96.55 |
| [nomtp-fp8kv-3x320K-text-image.env](inkling-2-mini/nomtp-fp8kv-3x320K-text-image.env) | 纯 AR | 3 × 320,000 | 1,006,906 | 2885.31 / 147.18 | 2588.91 / 127.83 |
| [nomtp-fp8kv-4x210K-text-image.env](inkling-2-mini/nomtp-fp8kv-4x210K-text-image.env) | 纯 AR | 4 × 210,000 | 888,054 | 2952.78 / 186.48 | 2617.41 / 155.75 |
| [dflash2-fp8kv-1x870K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x870K-text-image.env) | DFlash2/7 | 1 × 870,000 | 926,614 | 2793.98 / 275.03 | 2580.39 / 207.40 |
| [dflash2-fp8kv-2x410K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x410K-text-image.env) | DFlash2/7 | 2 × 410,000 | 866,951 | 2626.30 / 408.58 | 2379.89 / 300.10 |
| [dflash2-fp8kv-3x220K-text-image.env](inkling-2-mini/dflash2-fp8kv-3x220K-text-image.env) | DFlash2/7 | 3 × 220,000 | 677,026 | 2601.92 / 493.51 | 2376.21 / 298.19 |
| [dflash2-fp8kv-4x140K-text-image.env](inkling-2-mini/dflash2-fp8kv-4x140K-text-image.env) | DFlash2/7 | 4 × 140,000 | 573,531 | 2736.95 / 594.34 | 2427.26 / 369.16 |
| [dspark7-fp8kv-1x900K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x900K-text-image.env) | DSpark/7 | 1 × 900,000 | 930,394 | 2700.40 / 233.86 | 2468.55 / 185.43 |
| [dspark7-fp8kv-2x430K-text-image.env](inkling-2-mini/dspark7-fp8kv-2x430K-text-image.env) | DSpark/7 | 2 × 430,000 | 897,728 | 2575.34 / 354.87 | 2431.29 / 263.60 |
| [dspark7-fp8kv-3x270K-text-image.env](inkling-2-mini/dspark7-fp8kv-3x270K-text-image.env) | DSpark/7 | 3 × 270,000 | 848,950 | 1379.51 / 414.08 | 1331.07 / 282.97 |
| [dspark7-fp8kv-4x190K-text-image.env](inkling-2-mini/dspark7-fp8kv-4x190K-text-image.env) | DSpark/7 | 4 × 190,000 | 795,839 | 1706.33 / 491.95 | 1545.74 / 336.36 |

- 测试环境为 vLLM 2080 Ti Definitive Edition `v0.3.0`、上游 vLLM `v0.34.0`、PyTorch `2.14.0+cu130`、CUDA 13.0、NVIDIA 驱动 595.71.05。验证机使用 PCIe，关闭 NCCL P2P。从镜像提取的补丁运行时以原生进程运行，沿用 launcher 的 `fast` 参数。
- 每路请求含一张 480 × 480 图片，续写 `1 2 3 4`，关闭 Prefix Cache。两组测试都按表中并发执行，各取三轮中位数，单位为 tok/s。并发 prefill 计至最晚的首 token，并发 decode 取共同解码窗口的合计速度，窗口两端各裁 10%，每端最多 250 ms。测试期间零抢占。KV 容量来自启动日志，吞吐实测每路最长 32,768 输入 token。
- DFlash2 使用 FP16 草稿 KV，本次测速请求逐条报告的草稿接受率为 100%。DSpark 从主权重加载 `mtp.safetensors`。
