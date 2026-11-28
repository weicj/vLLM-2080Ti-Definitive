# 2xT10 Profiles

语言：[English](README.md) | 简体中文

## Qwen3.8-27B-INT4

权重 [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [mtp3-fp8kv-1x262K-text-only.env](qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env) | 262,144 | FP8 | MTP/3 | text-only | 293,422 | 1176.34 / 87.28 | 1107.18 / 81.90 |
| [mtp3-tqk8v4-2x155K-text-only.env](qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env) | 2 × 155,648 | TQK8V4 | MTP/3 | text-only | 351,112 | 1059.92 / 95.09 | 1069.71 / 107.12 |
| [mtp3-tq4nc-2x220K-text-only.env](qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env) | 2 × 220,000 | TQ4NC | MTP/3 | text-only | 511,566 | 1164.18 / 85.29 | 1122.17 / 73.27 |
| [mtp3-tqk8v4-1x196K-text-image.env](qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env) | 196,000 | TQK8V4 | MTP/3 | text+image | 253,866 | 1196.41 / 87.30 | 1136.75 / 60.36 |

## Inkling 2 Mini

权重与草稿 [官方权重](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16 草稿](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

所有档使用 FP8 主模型 KV。DSpark/7 自动加载官方权重里的 `mtp.safetensors`。DFlash2/7 使用 FP16 草稿 KV 和 LBNHC 布局。

| Profile | 并发 × 每路上下文 | 测速并发 | 启动 KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| [dspark7-fp8kv-2x200K-text-only.env](inkling-2-mini/dspark7-fp8kv-2x200K-text-only.env) | 2 × 200,000 | 2 | 411,276 | 2003.21 / 338.11 | 1911.51 / 202.07 |
| [dspark7-fp8kv-1x262K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x262K-text-image.env) | 1 × 262,144 | 1 | 306,544 | 2480.39 / 249.43 | 2419.47 / 165.72 |
| [dflash2-fp8kv-1x262K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x262K-text-image.env) | 1 × 262,144 | 1 | 317,129 | 2465.67 / 237.90 | 2418.85 / 174.58 |
| [dflash2-fp8kv-2x128K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x128K-text-image.env) | 2 × 128,000 | 2 | 267,636 | 2009.46 / 376.91 | 2113.50 / 236.45 |
| [dflash2-fp8kv-2x176K-text-only.env](inkling-2-mini/dflash2-fp8kv-2x176K-text-only.env) | 2 × 176,000 | 2 | 369,278 | 2370.57 / 384.04 | 2108.90 / 238.03 |
| [nomtp-fp8kv-1x522K-text-image.env](inkling-2-mini/nomtp-fp8kv-1x522K-text-image.env) | 1 × 522,000 | 1 | 523,533 | 3217.41 / 48.47 | 2858.22 / 46.10 |
| [nomtp-fp8kv-2x203K-text-image.env](inkling-2-mini/nomtp-fp8kv-2x203K-text-image.env) | 2 × 203,000 | 2 | 409,366 | 1923.62 / 87.11 | 2257.81 / 78.92 |

双卡 DSpark/7 图文启动示例

```bash
MODEL_DIR=/path/to/Inkling-2-Mini \
PROFILE=2xT10/inkling-2-mini/dspark7-fp8kv-1x262K-text-image.env \
GPU_DEVICES=0,1 TP_SIZE=2 NCCL_P2P_DISABLE=0 DISABLE_PREFIX_CACHING=1 \
NON_INTERACTIVE=1 ./launcher.sh
```

## 测试说明

- **环境**为vLLM 2080 Ti Definitive Edition `v0.3.0`，基于官方 vLLM `v0.34.0` Docker 镜像。CUDA 13.0、PyTorch `2.14.0+cu130`、NVIDIA 驱动 610.43.03。2 张 16 GiB Tesla T10，TP2、PCIe P2P，使用 fast 模式并关闭 Prefix Cache。Inkling 使用 NCCL，纯 AR 同时启用 vLLM custom all-reduce。测试日期为 2026-11-27 至 28 日。
- **吞吐**单位为 tok/s，4K/128 与 32K/512 分别是每路 4,096 输入 / 128 输出和 32,768 输入 / 512 输出。并发 prefill 以最晚首 token 为止，并发 decode 只统计共同解码窗口，窗口两端各裁 10%，每端最多 250 ms。
- **Qwen** 用单路纯文本，批处理上限为 2,048 token。4K/128 取三次中位数，32K/512 跑一次。196K 档使用 `1 2 3 4` 续写，草稿接受率 100%，其余使用固定 token ` the`。FP8 KV 262K 和 TQK8V4 155K 档跑通了满上下文。
- **Inkling** 的 DSpark/7 和 DFlash2/7 使用 `1 2 3 4` 高接受率续写，图文请求各含一张 480 × 480 图片。4K/128 取三轮中位数。32K/512 的 DSpark/7 取两轮，DFlash2/7 取三轮。DSpark/7 的 1 × 262K 图文档另行跑通满上下文。DFlash2/7 的草稿接受率为 100%，零抢占。
- **纯 AR** 使用 LBNHC 布局和 97% 显存利用率。每路上下文按启动 KV 容量配置并留少量余量。4K/128 和 32K/512 均在表中并发下取三轮中位数，每条请求带一张图。
