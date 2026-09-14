# Profile 导引

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设。硬件专用 profile 还可设置 `TP_SIZE`；
checkpoint、GPU、端口、chat template 和 reasoning 默认值仍由 launcher 管理，模型路径
通过 `MODEL_DIR` 单独指定。

当前目录按 `硬件 / 模型 / 权重 / 启动模式` 组织：

```text
profiles/
  2x2080Ti/
    qwen27b/
      w8a16/               # Qwen3.8 FP8 权重
      w4a16/               # Qwen3.8 NVFP4 权重
    qwen35b/
      w8a16/               # Qwen3.x 35B FP8 权重
  4xT10/
    qwen27b/
      w8a16/               # Qwen3.8 FP8 权重，TP=4 且启用 CAR
```

`w8a16` 表示 FP8 权重、FP16 激活；`w4a16` 表示 NVFP4 权重、FP16 激活。
`normal`、`fast`、`aggressive` 是由 `COMPATIBLE_MODES` 表示的启动模式。
`MTP_K=0` 表示不使用 MTP，`MTP_K=3` 表示 MTP3。所有 shipped profile 都显式写有
`MESSAGE_TYPE=text-only` 或 `MESSAGE_TYPE=text+image`。

## Profile 与参考性能

<small>pre3 历史参考环境：`.31`，双路 Intel Xeon E5-2673 v4（40 核 / 80 线程）、60 GiB
内存 + 8 GiB swap，双 RTX 2080 Ti 22 GiB、NVLink（SM75），Ubuntu 26.04、kernel
7.0.0-30、驱动 595.84、CUDA 13.0、PyTorch 2.13、vLLM 0.27.1、TP=2/PP=1，启用
CUDA Graph 且未启用 eager。性能统一按 4K 输入 / 128 输出，表示 prefill / decode
tok/s。这些数据不能验证采用 upstream nightly `b23433088b` 的 pre4。</small> `-` 表示没有稳定测量值。详细证据见
[历史验证记录](../docs/2080ti-0.2.1-pre-validation.md)。

下方 4xT10 行在四张 16 GiB Tesla T10（PCIe、TP=4）上测得，使用 pre4
`315d5930f5` 加上 [#153](https://github.com/weicj/vLLM-2080Ti-Definitive/pull/153)
的 ABI 匹配 PCIe CAR 扩展（`a910fe9e43`）。它们依赖该扩展，未打补丁的 pre4
不是等价路线。两组运行均保留 custom all-reduce，日志均为 `['CUSTOM', 'PYNCCL']`，
通过启动和图像请求；性能取三个不同 prompt 的 4K/128 独立请求中位数。fast TQK8V4
路线在 SM75 上使用 FlashInfer FA2，完成 FULL 和 PIECEWISE CUDA Graph capture。
固定 32 题 GSM8K smoke 中，normal FP16 为 21/32，fast TQK8V4 为 26/32；28,828-token
检索、多轮、图像 OCR 和 WAL 语义探针也均通过。图像 smoke 仅验证视觉请求路由和生成，
不代表视觉回答质量已完成评估。

### [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-104K-mtp3-text-image.env` | normal | 104K | FP16 | 3 | text+image | 110,784 | 1506.86 / 82.88 |
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 138,394 | 1496.95 / 83.90 |
| `2x2080Ti/qwen27b/w8a16/normal/fp16kv-144K-nomtp-text-only.env` | normal | 144K | FP16 | 0 | text-only | 152,749 | 1501.39 / 30.40 |
| `2x2080Ti/qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 310,827 | 1525.37 / 83.51 |
| `2x2080Ti/qwen27b/w8a16/normal/fp8kv-220K-mtp3-text-image.env` | normal | 220K | FP8 | 3 | text+image | 231,169 | 1555.1 / 70.1 |
| `2x2080Ti/qwen27b/w8a16/normal/fp8kv-256K-mtp3-text-only.env` | normal | 256K | FP8 | 3 | text-only | 320,232 | 1573.95 / 67.58 |
| `4xT10/qwen27b/w8a16/normal/fp16kv-256K-mtp3-text-image.env` | normal | 256K | FP16 | 3 | text+image | 312,585 | 1440.31 / 73.68 |
| `4xT10/qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-image.env` | fast | 256K | TQK8V4 | 3 | text+image | 780,814 | 1692.12 / 104.78 |

### [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-only.env` | normal | 240K | FP8 | 3 | text-only | 463,890 | 1433.2 / 76.8 |
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-240K-mtp3-text-image.env` | normal | 240K | FP8 | 3 | text+image | 426,080 | 1250.6 / 52.5 |
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 | 0 | text-only | 518,191 | 1372.1 / 42.0 |
| `2x2080Ti/qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC | 3 | text-only | 732,381 | 1402.9 / 103.5 |

### 并发测试路线（NVFP4 纯文本）

| Profile | 模式 | 上下文 | KV/MTP | GPU KV tokens | C1 | C2 | C4 | C8 | 证据 |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `2x2080Ti/qwen27b/w4a16/normal/fp8kv-192K-nomtp-text-only.env` | normal | 192K | FP8 / 0 | 518,191 | 1372.1 / 42.0 | 1507.4 / 80.4 | 1535.9 / 152.0 | 1523.4 / 270.8 | 完整窗口正式测试 |
| `2x2080Ti/qwen27b/w4a16/fast/tq4nc-262K-mtp3-text-only.env` | fast | 262K | TQ4NC / 3 | 732,381 | 1402.9 / 103.5 | 1449.1 / 180.4 | 1460.0 / 220.7 | 1449.1 / 347.3 | 完整窗口正式测试 |

每个 C 单元格均为 `prefill / 完整窗口 aggregate decode` tok/s，并已关闭 prefix cache。

### [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

| Profile | 启动模式 | 上下文 | KV | MTP | 消息 | GPU KV tokens | 性能 |
|---|---|---:|---|---:|---|---:|---:|
| `2x2080Ti/qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 273,586 | 7378 / 128.7 |
| `2x2080Ti/qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 146,485 | 5965.8 / 127.6 |

选定 profile 后，启动服务前执行 `./launcher.sh --print-config` 检查最终生效的路线参数。
