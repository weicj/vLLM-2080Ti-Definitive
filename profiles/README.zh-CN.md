# Profile 指南

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设，不负责选择 checkpoint、GPU、端口、chat
template 或 reasoning 默认值；target 权重通过 `MODEL_DIR` 选择，DFlash draft
权重通过 `SPECULATIVE_MODEL` 单独选择。

Profile 按硬件、模型和权重格式分组。每个硬件/模型/权重目录下的文件直接平铺；
启动模式由 launcher 选择，默认使用 `fast`。

```text
profiles/
  2x2080Ti/   # [详细 Profile 说明与参考性能](2x2080Ti/README.zh-CN.md)
  4xT10/      # [详细 Profile 说明与参考性能](4xT10/README.zh-CN.md)
  2xT10/      # [详细 Profile 说明与参考性能](2xT10/README.zh-CN.md)
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

```dotenv
MODEL_FAMILY=qwen35
MODEL_VARIANT=fp8
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8
MAX_MODEL_LEN=262144
GPU_UTIL=0.96
MAX_NUM_SEQS=1
SPECULATIVE_METHOD=mtp
SPECULATIVE_TOKENS=4
MESSAGE_TYPE=text-only
```

文件名和字段值必须保持一致：`nomtp`、`mtpN`、`dflash` 要分别匹配投机方法和
token 数；文件名中的并发数和 `K` 形式的上下文要分别匹配 `MAX_NUM_SEQS` 和
`MAX_MODEL_LEN`；KV 与消息类型后缀也要匹配对应字段。手写完成后，先运行
`bash tools/validate_profiles.sh` 校验，再使用该 Profile。

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

### Qwen3.8 Flash-Next EXL3 TP2xPP2

`qwen38flashnext/exl3/experimental/tp2pp2-ssd-nomtp-text.env` 是四张 T10
上的首个 EXL3 功能验证路线。它使用 TP=2、PP=2 与 expert parallel，使每个本地
expert 保留 640 宽 intermediate dimension，并复用 Qwen4Exp Flash-Next 的 PP
实现，由 launcher 自动打开 SSD PLE n-gram streaming。启动前请按
[`docs/usage/exl3_turing.md`](../docs/usage/exl3_turing.md) 安装固定版本的
`vllm-exl3-turing` 与 `exllamav3-turing`。在 TP2xPP2 通过 CUDA Graph、输出
一致性和 EXL3 loader 路径检查前，不记录吞吐数字。
选定 profile 后，启动服务前执行 `./launcher.sh --print-config` 检查最终生效的路线参数。
