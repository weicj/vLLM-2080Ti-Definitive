# Profile 导引

语言：[English](README.md) | 简体中文

这里是 vLLM 2080Ti Definitive 自带的启动 profile。一个 profile 只是运行参数
的 `.env` 预设，不包含模型权重路径；权重目录通过 `launcher.sh` 或
`MODEL_DIR=...` 单独选择。

这里列出的上下文容量和吞吐数据，验证硬件是双 RTX 2080 Ti 22GB，tensor
parallel size 2。

目录结构：

```text
profiles/
  templates/
  qwen27b/
    normal/
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
  qwen35b/
    normal/
      fp8/
    aggressive/
      fp8/
    fast/
      fp8/
    user/
```

启动模式：

- `safe`：保守回退模式，优先保证可用性。
- `normal`：推荐日常模式，适合稳定部署。
- `fast`：高性能模式，适合追求更高吞吐的场景。
- `aggressive`：更加激进的模式，性能与质量风险最高。

`profiles/templates/` 存放可选 chat template 预设。它们通过 launcher 作为全局
服务设置选择；具体 route profile 不保存 chat template、GPU、端口、reasoning
默认值或工具调用默认值。
对内置 Qwen3/Qwen3.x 路线，launcher 会在未显式设置时补上 `qwen3`
reasoning parser，让启动 smoke 和聊天解析都跟模型默认 reasoning 行为保持一致。
如需诊断无 reasoning parser 路径，可设置 `REASONING_PARSER=off`。

文件名描述路线：

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV 精度定位：

- `fp16kv`：质量路线。
- `int8kv`：容量 / 平衡路线；当前只作为 `normal` profile 保留。
- `tqk8v4`：TurboQuant K8V4 压缩路线；当前只保留质量通过的 `fast` profile。
- 官方 Qwen3.x 35B 当前提供 FP8 权重 + FP16 KV 的纯文本和图文预设。

内置 TQK8V4 profile 使用 `MAX_BATCHED_TOKENS=2560`，这是 Qwen hybrid cache
block 对齐后的 prefix-cache 路径已验证设置。

## 已验证 Profile

### Qwen3.x 27B

测试权重：官方 [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)、
[Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)，以及
Jackrong/Qwopus3.6-27B-v2-FP8（后者约 29G）。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/fp16kv-138K-mtp3-text-only.env` | normal | 138K | FP16 | 3 | text-only | 1 | - [^qwen27b-fp8-138k] |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

[^qwen27b-fp8-138k]: 该路由为 MTP3 138K 部署跟踪；本 PR 未单独跑吞吐基准（最接近的已测 qwen27b FP8 MTP3 路由为 `fp16kv-128K`，1619.48 / 84.71）。

### Qwen3.x 35B

测试目标权重：Qwen/Qwen3.6-35B-A3B-FP8，约 36G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.x 27B

测试权重：QuantTrio/Qwen3.6-27B-AWQ、mconcat/Qwopus3.6-27B-v2-AWQ-4bit，以及
llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4，
约 19G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 每工作区 250K | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 每工作区 250K | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

### Qwen3.8-27B（INT4）

测试权重：[SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)，
18.22 GB，GPTQ INT4 G128，MTP 以 BF16 保留。

**关键发现**：Qwen3.8 GPTQ INT4 + FP16 KV + MTP3 在 4K PP/TG 下 decode 达 **126.2 tok/s**，相较
Qwen3.6-27B（4K PP/TG 下 81.7 tok/s）**+54%**，二者采用同一 4K PP/TG 基准口径。逐配置对比见基准文档。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---|---:|---|---:|---|---:|---:|
| `qwen38/fast/int4/fp16kv-128K-mtp3-text-only.env` | fast | 128K | FP16 | 3 | text-only | 1 | **1690.7 / 126.2** |
| `qwen38/normal/int4/int8kv-128K-nomtp-text-only.env` | normal | 128K | INT8 | 0 | text-only | 1 | 1741.4 / 54.7 |

详细基准结果见 [docs/benchmarks/qwen38-gptq-int4-benchmark.md](../docs/benchmarks/qwen38-gptq-int4-benchmark.md)。
