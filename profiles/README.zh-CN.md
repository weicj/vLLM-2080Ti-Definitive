# Profile 导引

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设，不负责选择 checkpoint、GPU、端口、chat
template 或 reasoning 默认值；模型路径通过 `MODEL_DIR` 单独指定。

当前目录按 `模型 / 权重 / 启动模式` 组织：

```text
profiles/
  qwen27b/
    w8a16/                 # Qwen3.8 FP8 权重
      normal/
      fast/
    w4a16/                 # Qwen3.8 NVFP4 权重
      normal/
      fast/
  qwen35b/
    w8a16/                 # Qwen3.x 35B FP8 权重
      normal/
```

`w8a16` 表示 FP8 权重、FP16 激活；`w4a16` 表示 NVFP4 权重、FP16 激活。
`normal`、`fast`、`aggressive` 是由 `COMPATIBLE_MODES` 表示的启动模式。
`MTP_K=0` 表示不使用 MTP，`MTP_K=3` 表示 MTP3。所有 shipped profile 都显式写有
`MESSAGE_TYPE=text-only` 或 `MESSAGE_TYPE=text+image`。

## Profile 与参考性能

<small>参考环境：`.31`，双路 Intel Xeon E5-2673 v4（40 核 / 80 线程）、60 GiB
内存 + 8 GiB swap，双 RTX 2080 Ti 22 GiB、NVLink（SM75），Ubuntu 26.04、kernel
7.0.0-30、驱动 595.84、CUDA 13.0、PyTorch 2.13、vLLM 0.27.1、TP=2/PP=1，启用
CUDA Graph 且未启用 eager。性能统一按 4K 输入 / 128 输出，表示 prefill / decode
tok/s。</small> `-` 表示没有稳定测量值。详细证据见
[`docs/2080ti-0.2.1-pre-validation.md`](../docs/2080ti-0.2.1-pre-validation.md)。

### [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w8a16/normal/fp16kv-104K-mtp3-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `qwen27b/w8a16/normal/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `qwen27b/w8a16/normal/fp8kv-220K-mtp3-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `qwen27b/w8a16/normal/fp8kv-256K-mtp3-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |

### [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env` | normal | 240K | FP8 | 3 | text-only | 463,890 | 1433.2 / 76.8 |
| `qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-image.env` | normal | 240K | FP8 | 3 | text+image | 426,080 | 1250.6 / 52.5 |
| `qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 | 0 | text-only | 518,191 | 1372.1 / 42.0 |
| `qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC | 3 | text-only | 732,381 | 1402.9 / 103.5 |

### 并发测试路线（NVFP4 纯文本）

| Profile | 模式 | 上下文 | KV/MTP | GPU KV tokens | C1 | C2 | C4 | C8 | 证据 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 / 0 | 518,191 | 1372.1 / 42.0 | 1507.4 / 80.4 | 1535.9 / 152.0 | 1523.4 / 270.8 | 完整窗口正式测试 |
| `qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC / 3 | 732,381 | 1402.9 / 103.5 | 1449.1 / 180.4 | 1460.0 / 220.7 | 1449.1 / 347.3 | 完整窗口正式测试 |

每个 C 单元格均为 `prefill / 完整窗口 aggregate decode` tok/s，并已关闭 prefix cache。

### [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 273,586 | 7378 / 128.7 |
| `qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 146,485 | 5965.8 / 127.6 |

### Qwen3.8 Flash-Next NVFP4 实验性 PP 路线

模型：[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)。
这是 SM75 上的实验性功能验证路线：NVFP4/Marlin、磁盘 PLE、FP16 KV、无 MTP、非 eager
CUDA Graph。以下为仓库统一口径的固定 4K/128 形状中位数，仅用于对齐比较；受多卡通信复杂度
和 CPU 能力限制，真实工作负载性能可能降到个位数 tok/s。

| Profile | TP/PP | 后端 | 上下文 / util | GPU KV tokens | Prefill / decode tok/s | 路线定位 |
|---|---:|---|---:|---:|---:|---|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 2x4 | FlashQLA + PYNCCL | 100K / 0.92 | 102,591 | **2379.14 / 24.92** | 推荐均衡路线 |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 4x2 | FlashQLA + PYNCCL | 90K / 0.96 | 100,031 | **1697.99 / 28.18** | Decode 优先 |
| TP2xPP4 异构容量参考 | 2x4 | FlashQLA + PYNCCL | 100K / 0.92 | **152,492** | — | 6 张 T10 + 2 张 RTX 2080 Ti；仅作启动/容量参考 |

TP2xPP4 是推荐的均衡路线；TP4xPP2 偏向 decode。CAR `auto` 仅用于 TP 组全量 NVLink 互联且 P2P 可达的情况，
PCIe-only 组使用 NCCL/PYNCCL。PP stage 之间可混用显卡型号，但每个 TP 组必须 P2P 有效。
真实 Agent/多轮性能和输出质量都应单独验证；启动前执行 `./launcher.sh --print-config`。

两者均为实验性工程路线；TP2xPP2 不作为已验证 profile 发布。

选定 profile 后，启动服务前执行 `./launcher.sh --print-config` 检查最终生效的路线参数。

异构 152,492-token 行使用 NVFP4 权重、FP16 KV、MTP=0、100K 上下文和非 eager CUDA Graph，
测量方式是启动/健康检查/真实请求容量探测，不是吞吐测试。
