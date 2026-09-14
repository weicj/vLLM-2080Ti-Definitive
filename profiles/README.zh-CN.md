# Profile 指南

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设。硬件专用 profile 还可固定已验证的
`TP_SIZE`、`PP_SIZE` 和 `VLLM_PP_LAYER_PARTITION` 拓扑；checkpoint、GPU、端口、
chat template 和 reasoning 默认值仍由 launcher 管理，模型路径通过 `MODEL_DIR` 单独指定。

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

Profile 文件名统一使用 `<解码类型>-<KV精度>-<并发数><上下文>-<消息类型>.env`。
例如 `dflash2-tqk8v4-4x220K-text-only.env` 表示 DFlash2、TQK8V4 KV、四并发、
每路 220K（十进制 token）上下文、纯文本消息。文件名中的上下文标签取
`MAX_MODEL_LEN / 1000` 的整数部分，例如 262144 标记为 262K。投机解码通过
`SPECULATIVE_METHOD=none|mtp|dflash` 路由，`SPECULATIVE_TOKENS` 默认分别为
`0`、`3`、`7`。逐请求投机指标由 launcher 设置：
`PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`。

手写 Profile 时，每行使用一个 `KEY=value` 赋值。必需的路线字段为
`MODEL_FAMILY`、`MODEL_VARIANT`、`QUANTIZATION`、`KV_CACHE_DTYPE`、
`MAX_MODEL_LEN`、`GPU_UTIL`、`MAX_NUM_SEQS`、`MESSAGE_TYPE`、
`SPECULATIVE_METHOD` 和 `SPECULATIVE_TOKENS`。可选路线字段为 `MODE`、
`MAX_BATCHED_TOKENS` 和 `ENABLE_YARN`。

例如，将下面内容保存为
`qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env`：

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
