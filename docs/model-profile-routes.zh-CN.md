# 模型 Profile 路线

本文记录双 RTX 2080 Ti / SM75 runtime 的模型部署 Profile 选择。限制条件放在
Profile 矩阵前面，因为它决定每条路线适合用在哪里。

吞吐细节见
[Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
[MTP 任务敏感性](mtp-task-sensitivity.md)。

## Qwen3.6 27B

### 限制条件

- INT8 KV 是纯文本容量路线，不推荐用于图像多模态。已验证的多模态 INT8
  路线可以 READY，但输出会退化成标点或异常文本重复，而不是稳定图像回答。
- FP16/default KV 只有 noMTP 模式真实通过了 `PP262000/TG1`。MTP3 的 262K
  服务可以 READY，但真实 262K prompt 会在 prefill 阶段 OOM，所以 FP16 下
  MTP3 仍然只作为短上下文速度路线。
- 多工作区 Profile 用于排队式工作区隔离，不代表真正并行长 prefill 吞吐；
  在这个 TP=2 runtime 下，重型长上下文任务实际仍会被串行化。
- YaRN 524K 是离线容量 Profile。正常低延迟交互服务仍以原生 262K 路线为默认。
- FP8 是权重路线，不是 KV 精度路线。SM75 上走的是 weight-only FP8，不是
  原生 FP8 tensor-core compute；因此它是最高质量的实用 8bit Qwen 路线，
  AWQ/GPTQ-INT4 仍然是默认性能 / 容量路线。
- 更大的 MTP 可以在纯吞吐测试中得到更高数字。MTP3 是接受率和真实任务总体
  吞吐率更平衡的部署参考。

### Profile 矩阵

| 使用场景 | 权重量化 | KV 精度 | 上下文大小 | 投机解码 | 消息类型 | 并发上限 |
|---|---|---|---|---|---|---|
| 最高质量 8bit 文本路线 | FP8 | FP16 | 已验证 8K-64K | Native MTP3 | 纯文本 | 1 请求 |
| 高质量原生上下文路线 | AWQ/GPTQ-INT4 | FP16 | 原生 262K | 无 | 纯文本 | 1 请求 |
| 短上下文峰值速度路线 | AWQ/GPTQ-INT4 | FP16 | 8K-16K | Native MTP3 | 纯文本 | 1 请求 |
| 高压缩路线 | AWQ/GPTQ-INT4 | TQ4NC | 原生 262K | Native MTP3 | 纯文本 | 1 请求 / 排队 |
| 超长上下文 | AWQ/GPTQ-INT4 | INT8 | 524K YaRN | Native MTP3 | 纯文本 | 1 离线请求 |
| 多工作区 | AWQ/GPTQ-INT4 | INT8 或 TQ4NC | 64K-262K caps | Native MTP3 | 纯文本 | 4 x 64K 排队 / 2 x 262K 排队 |
| 多模态 | AWQ/GPTQ-INT4 | TQ4NC | 原生 262K | Native MTP3 | 文本 + 图像 | 1 请求 |

### Launcher 预设

- `qwen27-awq-mtp3`：常规 Qwen 系 FP16/default KV + Native MTP3 路线。
- `qwen27-gptq-mtp3`：Qwen 系 GPTQ-INT4 FP16/default KV + Native MTP3 路线。
- `qwen27-fp8-mtp3`：Qwen 系 FP8 FP16/default KV + Native MTP3 路线。
- `qwen27-awq-mtp3-peak`：短上下文峰值速度文本路线。
- `qwen27-awq-mtp3-tq4nc-fi`：TQ4NC 压缩容量路线。
- `qwen27-awq-mtp3-int8kv`：INT8 KV 容量 / YaRN 路线。
- `qwen27-awq-mm-*` 和 `heretic27-gptq-mm-*`：Qwen 系模型变体的图像多模态
  实验预设。

## Gemma4 31B

### 限制条件

- Gemma4 是 head_dim=512 且带异构/GQA attention。在 SM75 上，压缩 KV 路线
  暂时没有已验证的 FlashAttention/FlashInfer 快速 prefill；262K INT8 目前
  回落到 SDPA/GQA，明显慢于 FP16/default KV。
- 在同样双 22GB 显存下，Gemma 的实际可用上下文明显小于 Qwen。head_dim=512、
  GQA/异构 attention，以及压缩 KV 分组效率不足，会让“能启动压缩 KV”不等于
  “能得到高效的 262K 满血服务 Profile”。
- FP16/default KV 是速度和质量最好的路线，但双 22GB 卡无法支撑完整原生
  262K 上下文。已验证服务 Profile 是 16K；`105216` 文本和 `97152` 图像
  只是 262K 失败探针里的启动估算值，不是真实请求通过的实用上限。
- TQ4NC 保留为快速压缩短上下文路线：真实 prompt 实用边界约 `43K`，
  `43005`-token prompt 通过，`43505`/`44005` admission 失败。`64K` 只是
  READY 证据，不是真实长 prompt 通过证据。
- Gemma 多模态保留在 default KV。INT8/TQ4NC 多模态不推荐，因为异构 head
  的多模态后端会拒绝或破坏这些压缩 KV 路线。

### Profile 矩阵

| 使用场景 | 权重量化 | KV 精度 | 上下文大小 | 投机解码 | 消息类型 | 并发上限 |
|---|---|---|---|---|---|---|
| 高质量路线 | GPTQ-INT4 | FP16 | 16K 已验证文本服务；105K 仅估算 | Assistant MTP5 | 纯文本 | 1 请求 |
| 快速压缩路线 | GPTQ-INT4 | TQ4NC | 43K 真实 prompt 实用边界 | 无 | 纯文本 | 1 请求 |
| 长上下文路线 | GPTQ-INT4 | INT8 | 原生 262K，慢速离线 | 无 | 纯文本 | 1 个慢速离线请求 |
| 多模态兼容 | GPTQ-INT4 | FP16 | 8K 图像已验证 | Assistant MTP3 兼容 | 文本 + 图像 | 1 请求 |

### Launcher 预设

- `gemma4-gptq-tq4nc-mtp3`：TQ4NC + assistant MTP3 兼容路线。
- `gemma4-gptq-tq4nc-nomtp`：TQ4NC no-MTP 短上下文路线。
- `gemma4-gptq-mm-nomtp`：default-KV 图像多模态兼容路线。
- `gemma4-gptq-mm-mtp3`：default-KV + assistant MTP3 图像多模态路线。
