# Profile 指南

语言：[English](README.md) | 简体中文

Inkling 的 Profile 位于 `profiles/<硬件>/inkling-2-mini/<路线>.env`。
有多个权重版本的模型使用 `profiles/<硬件>/<模型>/<权重>/<路线>.env`。
模型权重、配置文件和吞吐数据见硬件指南。

- [2xT10](2xT10/README.zh-CN.md)
- [4xT10](4xT10/README.zh-CN.md)
- [2x2080Ti](2x2080Ti/README.zh-CN.md)

通过 `./launcher.sh` 选择模型、GPU 拓扑、端口、聊天模板、思考设置和模式，默认模式为
`fast`。主模型权重使用 `MODEL_DIR`，DFlash 草稿使用 `SPECULATIVE_MODEL`。
可用 `./launcher.sh --print-config` 预览生效参数。[非交互启动说明](../docs/non-interactive-launch.zh-CN.md)

在 launcher 中设置 `PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`，选择逐请求投机解码指标的详细程度。

## Profile 格式

文件名为 `<解码类型>-<KV 精度>-<并发数>x<每路上下文>-<消息类型>.env`，与配置值对应。
上下文按十进制千位标记，例如 `MAX_MODEL_LEN=262144` 写作 `262K`。
`SPECULATIVE_METHOD` 可选 `none`、`mtp`、`dflash` 或 `dspark`，草稿长度由
`SPECULATIVE_TOKENS` 设置。DSpark 从 Inkling 官方权重目录加载草稿。

每行写一个 `KEY=value`。必填字段为 `MODEL_FAMILY`、`MODEL_VARIANT`、
`QUANTIZATION`、`KV_CACHE_DTYPE`、`MAX_MODEL_LEN`、`GPU_UTIL`、
`MAX_NUM_SEQS`、`MESSAGE_TYPE`、`SPECULATIVE_METHOD` 和
`SPECULATIVE_TOKENS`。[Profile 示例](2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env)

运行 `bash tools/validate_profiles.sh` 校验本地 Profile。可选字段也由脚本列出。[校验脚本](../tools/validate_profiles.sh)
