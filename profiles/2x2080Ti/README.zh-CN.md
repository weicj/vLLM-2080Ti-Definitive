# 2x2080Ti Profile

## Qwen3.8-27B-FP8

测试权重：[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env` | 148K | FP16 | MTP/4 | text-only | 149,964 | 1643.37 / 97.55 | 1425.34 / 95.10 |
| `qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env` | 176K | FP16 | 无/自回归 | text-only | 179,940 | 1694.94 / 33.17 | 1483.04 / 30.80 |
| `qwen27b/w8a16/nomtp-fp16kv-1x121K-text-image.env` | 121K | FP16 | 无/自回归 | text+image | 122,608 | 1697.78 / 32.81 | 1447.04 / 30.65 |
| `qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env` | 262K | FP8 | MTP/4 | text-only | 295,455 | 1623.68 / 94.60 | 1393.94 / 92.47 |
| `qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env` | 186K | FP8 | MTP/4 | text+image | 190,540 | 1634.73 / 96.11 | 1384.95 / 93.17 |
| `qwen27b/w8a16/yarn-fp8kv-1x338K-text-only.env` | 338K | FP8 | 无 + YaRN | text-only | 354,143 | 1239.30 / 23.71 | 1338.48 / 17.91 |

## Qwen3.8-27B-NVFP4

测试权重：[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)；DFlash draft：[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env` | 262K | FP8 | DFlash2/7 | text+image | 285,081 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| `qwen27b/w4a16/mtp4-fp8kv-2x229K-text-only.env` | 2 x 229K | FP8 | MTP/4 | text-only | 468,978 | 1432.48 / 116.04 | 1247.12 / 109.72 |
| `qwen27b/w4a16/yarn-fp8kv-1x524K-text-only.env` | 524K | FP8 | 无 + YaRN | text-only | 588,863 | 1466.29 / 41.36 | 1274.26 / 37.22 |
| `qwen27b/w4a16/dflash-fp8kv-2x176K-text-only.env` | 2 x 176K | FP8 | DFlash/7 | text-only | 357,194 | 1412.94 / 223.08 | 1249.27 / 211.10 |
| `qwen27b/w4a16/mtp4-tq4nc-3x262K-text-only.env` | 3 x 262K | TQ4NC | MTP/4 | text-only | 837,832 | 1446.95 / 127.86 | 1265.68 / 68.64 |

## Qwen3.6-35B-A3B-FP8

测试权重：[Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env` | 262K | FP16 | 无/自回归 | text-only | 284,760 | 6690.07 / 113.63 | 5941.16 / 105.92 |
| `qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env` | 221K | FP8 | 无/自回归 | text+image | 223,158 | 6357.56 / 110.25 | 5493.10 / 102.23 |

## 说明

1. Profile 直接放在各模型/权重目录下，不再按 mode 分目录。Mode 由 launcher 选择，默认使用 `fast`。
2. 性能数据统一使用 launcher 的可复测参考口径：仅在测试期间关闭 Prefix Cache、单次只发送一个纯文本请求、预热不计入统计、4K/128 取三次中位数，并完整运行 32K/512。图文 Profile 同样使用纯文本性能口径，图像语义另行验证。`4K/128` 表示准确的 4,096 输入 token，`32K/512` 表示准确的 32,768 输入 token；原始日志和请求 JSON 保存在仓库外部的内部审计目录。
3. NVFP4 W4A16 的 FP16KV 存在显著质量塌陷问题，因此不列入推荐 KV 类型。
4. NVFP4 DFlash2 图文路线已使用明确的自然语言图像问题验证，正确返回“蓝色方形/橙色圆形”。由于 DFlash draft 不支持外部多模态 embedding，draft 侧使用纯文本输入；图像理解仍由 target model 完成。

5. 测试环境：2026-09-19，软件版本 v0.2.1。测试主机为双路 Intel Xeon E5-2673 v4（共 80 个逻辑 CPU），内存 60 GiB，Swap 8 GiB。测试拓扑使用物理 GPU 1 和 5，两张 NVIDIA GeForce RTX 2080 Ti（每张 22,528 MiB），两卡之间为 NV2 连接；服务使用 TP2。NVIDIA 驱动版本为 595.91.07。运行环境为仓库的 `vllm-sm75-tp2-cu130`（CUDA 13.0，torch 2.13.0+cu130）。
