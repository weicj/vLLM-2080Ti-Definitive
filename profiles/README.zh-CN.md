# Profile 导引

语言：[English](README.md) | 简体中文

Profile 是只保存路线参数的 `.env` 预设，不负责选择 checkpoint、GPU、端口、chat
template 或 reasoning 默认值；target 权重通过 `MODEL_DIR` 选择，DFlash draft
权重通过 `SPECULATIVE_MODEL` 单独选择。

Profile 首先按硬件分组，再按模型、权重格式和启动模式分层：

```text
profiles/
  2x2080Ti/   # [硬件说明](2x2080Ti/README.zh-CN.md)
  4xT10/      # [硬件说明](4xT10/README.zh-CN.md)
  8xT10/      # [硬件说明](8xT10/README.zh-CN.md)
```

Profile 文件名统一使用 `<解码类型>-<KV精度>-<并发数><上下文>-<消息类型>.env`。
例如 `dflash2-tqk8v4-2x172k-text-only.env` 表示 DFlash2、TQK8V4 KV、双并发、
每路 172K 上下文、纯文本消息。投机解码通过
`SPECULATIVE_METHOD=none|mtp|dflash` 路由，`SPECULATIVE_TOKENS` 默认分别为
`0`、`3`、`7`。
逐请求投机指标由 Launcher 设置：
`PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`；DFlash 默认使用
`detailed`，可在 Launcher 中切换。

选择 profile 后，可执行 `./launcher.sh --print-config` 检查最终生效的路线参数。

八张 T10 的 Qwen3.8 Flash-Next PP profile 说明见
[`8xT10/README.zh-CN.md`](8xT10/README.zh-CN.md)。在审计 runner 完成预热和三次正式
采样并写出 manifest 前，吞吐数据会保留为待测；历史数字不会未经复测直接晋升到路线清单。
