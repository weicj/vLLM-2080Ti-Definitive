# 4xT10 Profile

## Qwen3.8-27B-FP8

测试权重：[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)；DFlash2 draft：[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 300,304 | 1433.91 / 191.89 | 1536.13 / 189.38 |
| `qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env` | 240K | FP16 | DFlash2/7 | text+image | 241,215 | 1443.96 / 190.89 | 1532.96 / 187.85 |
| `qwen27b/w8a16/dflash2-fp8kv-2x220K-text-only.env` | 2 x 220K | FP8 | DFlash2/7 | text-only | 445,644 | 1395.16 / 200.56 | 1486.57 / 210.18 |
| `qwen27b/w8a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 320,484 | 1484.30 / 106.04 | 1527.71 / 104.25 |
| `qwen27b/w8a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 565,524 | 1459.21 / 102.41 | 1521.55 / 104.32 |

## Qwen3.8-27B-NVFP4

测试权重：[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)；DFlash2 draft：[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env` | 262K | FP16 | DFlash2/7 | text-only | 432,205 | 1435.68 / 220.99 | 1505.42 / 216.84 |
| `qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env` | 2 x 262K | FP8 | DFlash2/7 | text-only | 763,494 | 1334.35 / 213.94 | 1348.40 / 228.76 |
| `qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env` | 4 x 220K | TQK8V4 | DFlash2/7 | text-only | 923,137 | 1374.96 / 234.92 | 1380.19 / 154.43 |
| `qwen27b/w4a16/mtp4-fp16kv-1x262K-text-image.env` | 262K | FP16 | MTP/4 | text+image | 436,388 | 1526.32 / 125.70 | 1517.31 / 123.13 |
| `qwen27b/w4a16/mtp4-fp8kv-2x262K-text-image.env` | 2 x 262K | FP8 | MTP/4 | text+image | 823,249 | 1507.94 / 116.86 | 1477.49 / 118.77 |

## 说明

1. Profile 直接放在各模型/权重目录下，不再按 mode 分目录。Mode 由 launcher 选择，默认使用 `fast`。
2. 性能数据统一使用 launcher 的可复测参考口径：仅在测试期间关闭 Prefix Cache、单次只发送一个纯文本请求、预热不计入统计、4K/128 取三次中位数，并完整运行 32K/512。图文 Profile 同样使用纯文本性能口径，图像语义另行验证。`4K/128` 表示准确的 4,096 输入 token，`32K/512` 表示准确的 32,768 输入 token。原始日志和请求 JSON 已留存备查。
3. 测试环境：FP8/W8A16 profile 于 2026-09-19 验证，NVFP4/W4A16 profile 于 2026-09-23 验证，软件版本均为 v0.2.1。测试主机为双路 Intel Xeon E5-2673 v4（共 80 个逻辑 CPU），内存 60 GiB，Swap 8 GiB。服务使用 TP4，运行在同一 NUMA 节点上通过 PCIe PIX 连接的四张 Tesla T10 上，每张卡有 16,384 MiB 显存。NVIDIA 驱动版本为 595.91.07。运行环境为仓库的 `vllm-def-cu130`（CUDA 13.0，torch 2.13.0+cu130）。
4. `4x220K` 路线固定使用 `GPU_UTIL=0.93`，为运行时临时显存预留空间。四路同时 32K 请求已通过；四路同时接近 220K 上限不属于最坏情况保证，DFlash 临时 buffer 可能导致 OOM。多并发 profile 不应同时接收多个超长请求。
5. 所有 profile 均使用 launcher 默认的 `MAX_BATCHED_TOKENS=2048`；FP8KV 路线生产环境应启用 Prefix Cache。上表性能测试特意关闭了 Prefix Cache。W8A16 FP8KV DFlash2 2x220K 路线依据 `GPU_UTIL=0.96` 下实测 445,644 GPU KV tokens 进行容量定标；双路同时 32K/512 验证均准确生成 512 tokens 且服务保持健康，并发时两路解码速度分别约为 62.88 和 65.70 tok/s。
