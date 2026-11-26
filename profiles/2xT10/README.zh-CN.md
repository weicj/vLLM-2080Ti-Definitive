# 2xT10 Profile

## Qwen3.8-27B-INT4

测试权重：[RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

### 0.3.x Docker 验证

2026-11-27，使用基于官方 vLLM v0.34.0 CUDA 13.0 镜像的 `v0.3.0.dev0`，
以 `fast` 模式运行原有 `qwen27b/w4a16/mtp3-fp8kv-1x262K-text-only.env`。
测试使用两张 16 GiB Tesla T10，位于同一 NUMA 节点，通过 PCIe PIX 连接。
NVIDIA 驱动版本为 610.43.03。

| 上下文 | GPU KV tokens | 4K/128 prefill / decode (tok/s) | 32K/512 prefill / decode (tok/s) |
|---|---:|---:|---:|
| 262,144 | 293,422 | 1176.34 / 87.28 | 1107.18 / 81.90 |

测速沿用下方参考口径，使用 FP8 KV、MTP/3，关闭 Prefix Cache，逐个发送纯文本请求。
同配置下，262,143 个输入 token 加 1 个输出 token 的容量请求耗时 476.5 秒。
启用 Prefix Cache 后，launcher 通过 smoke test，报告 288,504 个 GPU KV tokens。
重复请求命中了 4,096 个输入 token 中的 3,200 个，两次生成 token ID 一致。

### 0.2.x 原生验证

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
|---|---:|---|---|---|---:|---:|---:|
| `qwen27b/w4a16/mtp3-fp8kv-1x262K-text-only.env` | 262K | FP8 | MTP/3 | text-only | 266,537 | 1122.00 / 87.88 | 1070.10 / 83.57 |
| `qwen27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env` | 2 x 155K | TQK8V4 | MTP/3 | text-only | 321,789 | 1138.22 / 87.42 | 1000.18 / 59.55 |
| `qwen27b/w4a16/mtp3-tq4nc-2x220K-text-only.env` | 2 x 220K | TQ4NC | MTP/3 | text-only | 471,807 | 1134.92 / 83.54 | 1100.51 / 49.76 |
| `qwen27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env` | 196K | TQK8V4 | MTP/3 | text+image | 206,888 | 1138.74 / 87.46 | 1053.94 / 50.20 |

## 说明

1. Profile 直接平铺在模型/权重目录下；启动模式由 launcher 选择，默认使用 `fast`。
2. 性能数据统一使用 launcher 的可复测参考口径：仅在测试期间关闭 Prefix Cache、单次只发送一个纯文本请求、预热不计入统计、4K/128 取三次中位数，并完整运行 32K/512。图文 Profile 同样使用纯文本性能口径，图像语义另行验证。`4K/128` 表示准确的 4,096 输入 token 和 128 输出 token；`32K/512` 表示准确的 32,768 输入 token 和 512 输出 token。
3. 测试环境：2026-09-25 使用 v0.2.2 runtime development build 验证。测试主机为双路 Intel Xeon E5-2673 v4（共 80 个逻辑 CPU），内存 60 GiB，Swap 8 GiB。服务使用 TP2，运行在同一 NUMA 节点上通过 PCIe PIX 连接的两张 Tesla T10 上，每张卡有 16,384 MiB 显存。NVIDIA 驱动版本为 595.91.07。运行环境使用 CUDA 13.0 和 torch 2.13.0+cu130。
4. 多模态路线启用视觉分支后 196K 启动成功，并正确描述了 JPEG 流程图中的 proxy/prefill/decode 处理链路。
