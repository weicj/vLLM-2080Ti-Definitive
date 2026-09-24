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

选择 profile 后，可执行 `./launcher.sh --print-config` 检查最终生效的路线参数，
然后再启动服务。
