# 2xT10 Profiles

语言：[English](README.md) | 简体中文

## Qwen3.8-27B-INT4

权重 [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens | 4K/128 prefill / decode | 32K/512 prefill / decode |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| [`qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env`](qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env) | 262,144 | FP8 | MTP/3 | text-only | 293,422 | 1176.34 / 87.28 | 1107.18 / 81.90 |
| [`qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env`](qwen38-27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env) | 2 × 155,648 | TQK8V4 | MTP/3 | text-only | 351,112 | 1059.92 / 95.09 | 1069.71 / 107.12 |
| [`qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env`](qwen38-27b/w4a16/mtp3-tq4nc-2x220K-text-only.env) | 2 × 220,000 | TQ4NC | MTP/3 | text-only | 511,566 | 1164.18 / 85.29 | 1122.17 / 73.27 |
| [`qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env`](qwen38-27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env) | 196,000 | TQK8V4 | MTP/3 | text+image | 253,866 | 1191.44 / 82.76 | 1124.64 / 60.87 |

## Inkling 2 Mini

官方权重 [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

DSpark/7 从官方权重目录自动加载 `mtp.safetensors`，草稿采用概率采样。

| Profile | 上下文 | KV | 投机解码 | 消息 | GPU KV tokens |
| --- | ---: | --- | --- | --- | ---: |
| [`inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/nomtp-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | 纯 AR | text-only | 12,602 |
| [`inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env`](inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env) | 8,192 | FP8 | DSpark/7 | text-only | 10,833 |

| 提示词 | 纯 AR 4K/128 prefill / decode | DSpark/7 4K/128 prefill / decode | 草稿接受率 |
| --- | ---: | ---: | ---: |
| 自然语言 | 3644.31 / 63.67 | 3556.14 / 175.24 | 76.4–81.2% |
| 代码 | 3615.24 / 63.40 | 3540.27 / 183.17 | 86.5% |
| 重复文本 | 3673.90 / 62.98 | 3594.91 / 278.34 | 100% |

### DSpark/7 图文部署

| Profile | 并发 × 每路上下文 | GPU KV tokens | KV 显存 / 卡 | 4K/128 prefill / decode |
| --- | ---: | ---: | ---: | ---: |
| [inkling-2-mini/official/dspark7-fp8kv-2x12K-text-image.env](inkling-2-mini/official/dspark7-fp8kv-2x12K-text-image.env) | 2 × 12,000 | 28,675 | 4.21 GiB | 3063.20 / 92.81 |

同时发送两条各含一张 480 × 480 图片的请求，每条实测 11,872 输入 token（含图片与模板）加 128 输出 token。服务指标记录到两条请求同时运行，抢占为零。另用两张图片验证了蓝色方块、橙色圆形、绿色方块和紫色圆形的识别。

图文测试使用默认合并 Docker 镜像、FP8 KV、fast 模式和 NCCL PCIe P2P。视觉显存预热开启，Prefix Cache 关闭，GPU 显存利用率为 0.97，批处理上限为 1,024 token。吞吐为单路图文流式请求的 tok/s，4K/128 取三次中位数。vLLM 按当前上下文和 KV 分组计算等效容量，启动时报告 2.39 路满上下文并发。

## 容量与功能验证

FP8 KV 路线完成了 262,143 个输入 token 加 1 个输出 token 的容量请求，耗时 476.5 秒。
TQK8V4 路线完成了 155,647 个输入 token 加 1 个输出 token 的容量请求，耗时 205.4 秒。
FP8 KV 路线启用 Prefix Cache 后，launcher 通过 smoke test，报告 288,504 个 GPU KV tokens。
重复请求命中了 4,096 个输入 token 中的 3,200 个，两次生成 token ID 一致。
Docker 上的图文路线默认使用 PIECEWISE CUDA Graph，保留 MTP/3 后仍准确识别了
测试图卡上的蓝色方块与橙色圆形。

## 测试版本与环境

- 2026-11-27，vLLM 2080 Ti Definitive Edition `v0.3.0.dev0`。镜像基于官方 vLLM `v0.34.0`，使用 CUDA 13.0、PyTorch `2.14.0+cu130` 和 NVIDIA 驱动 610.43.03。
- 2 张 16 GiB Tesla T10，TP2，同一 NUMA 节点内 PCIe P2P。测试采用 `fast` 模式。纯文本吞吐测试的批处理上限为 2,048 token。Inkling 使用 NCCL PCIe P2P。
- 吞吐单位为 tok/s。纯文本吞吐测试预热后关闭 Prefix Cache，逐个发送请求。4K/128 取三次中位数，Qwen 的 32K/512 完整跑一次。4K/128 和 32K/512 分别是精确的 4,096 输入 / 128 输出和 32,768 输入 / 512 输出。图文 profile 的图像理解另行验证。
