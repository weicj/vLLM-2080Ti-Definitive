# 4xT10 Profiles

语言：[English](README.md) | 简体中文

## Qwen3.8-27B-FP8

权重 [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · DFlash2 草稿权重 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 测速并发 | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| [dflash2-fp16kv-1x262K-text-only.env](qwen38-27b/w8a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 331,998 | 1 | 1583.68 / 211.09 | 1466.17 / 210.14 |
| [dflash2-fp16kv-1x240K-text-image.env](qwen38-27b/w8a16/dflash2-fp16kv-1x240K-text-image.env) | 240,384 | FP16 | DFlash2/7 | text+image | 293,387 | 1 | 1584.73 / 209.94 | 1443.93 / 218.32 |
| [dflash2-fp8kv-2x220K-text-only.env](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 1 | 1340.62 / 210.43 | 1427.11 / 208.97 |
| [dflash2-fp8kv-2x220K-text-only.env](qwen38-27b/w8a16/dflash2-fp8kv-2x220K-text-only.env) | 2 × 220,000 | FP8 | DFlash2/7 | text-only | 515,625 | 2 | 1670.08 / 385.06 | 1487.07 / 355.14 |
| [mtp4-fp16kv-1x262K-text-image.env](qwen38-27b/w8a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 345,301 | 1 | 1578.10 / 121.59 | 1454.63 / 120.15 |
| [mtp4-fp8kv-2x262K-text-image.env](qwen38-27b/w8a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 650,056 | 1 | 1567.30 / 124.74 | 1428.35 / 125.80 |

## Qwen3.8-27B-NVFP4

权重 [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · DFlash2 草稿权重 [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 测速并发 | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| [dflash2-fp16kv-1x262K-text-only.env](qwen38-27b/w4a16/dflash2-fp16kv-1x262K-text-only.env) | 262,144 | FP16 | DFlash2/7 | text-only | 443,917 | 1 | 1559.39 / 249.82 | 1443.68 / 246.30 |
| [dflash2-fp8kv-2x262K-text-only.env](qwen38-27b/w4a16/dflash2-fp8kv-2x262K-text-only.env) | 2 × 262,144 | FP8 | DFlash2/7 | text-only | 703,905 | 1 | 1561.66 / 247.63 | 1428.82 / 243.43 |
| [dflash2-tqk8v4-4x220K-text-only.env](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 1 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| [dflash2-tqk8v4-4x220K-text-only.env](qwen38-27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env) | 4 × 220,000 | TQK8V4 | DFlash2/7 | text-only | 956,153 | 4 | 1585.96 / 797.18 | 1540.44 / 711.94 |
| [mtp4-fp16kv-1x262K-text-image.env](qwen38-27b/w4a16/mtp4-fp16kv-1x262K-text-image.env) | 262,144 | FP16 | MTP/4 | text+image | 451,426 | 1 | 1557.56 / 143.11 | 1437.54 / 137.96 |
| [mtp4-fp8kv-2x262K-text-image.env](qwen38-27b/w4a16/mtp4-fp8kv-2x262K-text-image.env) | 2 × 262,144 | FP8 | MTP/4 | text+image | 853,104 | 1 | 1547.72 / 141.89 | 1412.98 / 124.51 |

## Inkling 2 Mini

权重与草稿 [官方权重](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [DFlash2 BF16 草稿](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2)

所有档使用 FP8 主模型 KV。DSpark/7 自动加载官方权重里的 `mtp.safetensors`。DFlash2/7 使用 FP16 草稿 KV 和 LBNHC 布局。

| Profile | 并发 × 每路上下文 | 测速并发 | 启动 KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| [dspark7-fp8kv-1x1048K-text-image.env](inkling-2-mini/dspark7-fp8kv-1x1048K-text-image.env) | 1 × 1,048,576 | 1 | 1,252,517 | 3495.49 / 314.43 | 3517.83 / 232.72 |
| [dspark7-fp8kv-2x500K-text-image.env](inkling-2-mini/dspark7-fp8kv-2x500K-text-image.env) | 2 × 500,000 | 2 | 1,438,512 | 2014.68 / 405.64 | 2047.55 / 273.72 |
| [dspark7-fp8kv-4x250K-text-image.env](inkling-2-mini/dspark7-fp8kv-4x250K-text-image.env) | 4 × 250,000 | 4 | 1,256,622 | 1539.25 / 570.08 | 1288.98 / 372.64 |
| [dflash2-fp8kv-1x1048K-text-image.env](inkling-2-mini/dflash2-fp8kv-1x1048K-text-image.env) | 1 × 1,048,576 | 1 | 1,258,521 | 3729.50 / 290.63 | 3480.09 / 232.84 |
| [dflash2-fp8kv-2x512K-text-image.env](inkling-2-mini/dflash2-fp8kv-2x512K-text-image.env) | 2 × 512,000 | 2 | 1,083,356 | 3398.49 / 444.66 | 3353.87 / 338.78 |
| [dflash2-fp8kv-4x250K-text-image.env](inkling-2-mini/dflash2-fp8kv-4x250K-text-image.env) | 4 × 250,000 | 4 | 1,013,599 | 2966.63 / 653.72 | 2992.04 / 427.90 |

## 测试说明

- **环境**为vLLM 2080 Ti Definitive Edition `v0.3.0`，基于官方 vLLM `v0.34.0` Docker 镜像。CUDA 13.0、PyTorch `2.14.0+cu130`、NVIDIA 驱动 610.43.03。4 张 16 GiB Tesla T10，TP4、PCIe P2P，使用 fast 模式并关闭 Prefix Cache。Inkling 使用 NCCL。测试日期为 2026-11-27 至 28 日。
- **吞吐**单位为 tok/s，4K/128 与 32K/512 分别是每路 4,096 输入 / 128 输出和 32,768 输入 / 512 输出。并发 prefill 以最晚首 token 为止，并发 decode 只统计共同解码窗口，窗口两端各裁 10%，每端最多 250 ms。
- **Qwen** 用纯文本测速，批处理上限为 2,048 token。FP8 2 × 220K 使用 `1 2 3 4` 续写，其余使用固定 token ` the`。4K/128 取三轮中位数，32K/512 单路跑一次，并发取三轮。同一 PCIe switch 下的双路 FP8 和四路 NVFP4 测试均为 100% 草稿接受率、零抢占。NVFP4 2 × 262K 也跑通双路 32K/512。多路请求同时接近上下文上限时，DFlash 临时缓冲仍可能耗尽显存。
- **Inkling** 的 DSpark/7 和 DFlash2/7 用 `1 2 3 4` 高接受率续写，每路包含一张 480 × 480 图片。4K/128 取三轮中位数。32K/512 的 DSpark/7 取两轮，DFlash2/7 取三轮。四路 DFlash2 在同一 PCIe switch 下复测，批处理上限 1024，prefill 阈值 256。长上下文吞吐实测每路最多 32,768 输入 token，表中 KV 容量为启动估算。DFlash2/7 的草稿接受率为 100%，零抢占。
