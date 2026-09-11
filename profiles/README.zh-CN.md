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
  qwen38flashnext/
    w4a16/
      experimental/
```

启动模式：

- `safe`：保守回退模式，优先保证可用性。
- `normal`：推荐日常模式，适合稳定部署。
- `fast`：高性能模式，适合追求更高吞吐的场景。
- `aggressive`：更加激进的模式，性能与质量风险最高。

`profiles/templates/` 存放可选 chat template 预设。它们通过 launcher 作为全局
服务设置选择；具体 route profile 不保存 chat template、GPU、端口、reasoning
默认值或工具调用默认值。
`MODEL_FAMILY` 记录 checkpoint 架构而不是营销型号：`qwen35` 表示兼容
Qwen3.5 的 dense 架构，`qwen35moe` 表示对应 MoE 架构，`qwen4` 表示
Qwen4-Exp/Flash-Next。launcher 会读取 `config.json` 的 `model_type` 和
`architectures`，并据此过滤 profile。
对内置 Qwen3/Qwen3.6 路线，launcher 会在未显式设置时补上 `qwen3`
reasoning parser，让启动 smoke 和聊天解析都跟模型默认 reasoning 行为保持一致。
如需诊断无 reasoning parser 路径，可设置 `REASONING_PARSER=off`。

Qwen4 profile 可以设置 `PLE_PLACEMENT=disk|cpu|gpu`。`disk` 是已验证的默认
路线，由 CPU worker 直接按需读取 safetensors 映射；`cpu` 会在启动时把完整
PLE 表装入系统内存；`gpu` 会让完整 PLE 表常驻显存，必须确保单卡显存足够。
当前 T10 profile 使用 `disk`。

文件名描述路线：

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV 精度定位：

- `fp16kv`：质量路线。
- `int8kv`：容量 / 平衡路线；当前只作为 `normal` profile 保留。
- `tqk8v4`：TurboQuant K8V4 压缩路线；当前只保留质量通过的 `fast` profile。
- 官方 Qwen3.6 35B 当前提供 FP8 权重 + FP16 KV 的纯文本和图文预设。

内置 TQK8V4 profile 使用 `MAX_BATCHED_TOKENS=2560`，这是 Qwen hybrid cache
block 对齐后的 prefix-cache 路径已验证设置。

## 已验证 Profile

### Qwen3.6 27B FP8

测试权重：Jackrong/Qwopus3.6-27B-v2-FP8，约 29G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### Qwen3.6 35B FP8

测试目标权重：Qwen/Qwen3.6-35B-A3B-FP8，约 36G。

| Profile | 兼容模式 | 上下文 | KV | MTP | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen35b/normal/fp8/fp16kv-256K-nomtp-text-only.env` | normal | 256K | FP16 | 0 | text-only | 1 | 6705.13 / 97.33 |
| `qwen35b/normal/fp8/fp16kv-136K-nomtp-text-image.env` | normal | 136K | FP16 | 0 | text+image | 1 | 5485.13 / 95.20 |
| `qwen35b/aggressive/fp8/fp16kv-256K-nomtp-text-only.env` | aggressive | 256K | FP16 | 0 | text-only | 1 | 6843.01 / 124.01 |
| `qwen35b/aggressive/fp8/fp16kv-136K-nomtp-text-image.env` | aggressive | 136K | FP16 | 0 | text+image | 1 | 5422.83 / 124.11 |
| `qwen35b/fast/fp8/fp16kv-178K-mtp3-text-only.env` | fast | 178K | FP16 | 3 | text-only | 1 | 5889.20 / 195.95 |

### Qwen3.8 27B NVFP4 DFlash2

已验证目标权重：`nvidia/Qwen3.8-27B-NVFP4`；草稿权重：
`incoai/Qwen3.8-27B-DFlash2`。这是双 RTX 2080 Ti SM75 runtime 首条晋升的
DFlash2 路线。路线使用 TurboQuant K8V4 KV、K=7、B=8 分块的因果 target
验证、草稿侧 `TRITON_ATTN` 和 PIECEWISE CUDA Graph。吞吐采用高接受率固定
4K prompt / 128 completion，顺序为 prefill / decode tok/s。

| Profile | 兼容模式 | 上下文 | KV | DFlash K | 消息 | 并发 | 吞吐性能 |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env` | normal | 256K | TQK8V4 | 7 | text-only | 1 | **约 1450 / 162.83** |

证据：三次独立 4K/128 测试的 decode 分别为 155.16、166.57、166.76 tok/s，
平均 162.83 tok/s，均返回 128/128 token 并正常结束 stream。三次 4K 正确性
请求均精确返回 `PROFILE_OK`；服务以 262,144 上下文启动时暴露 331,935 个
KV token。该 profile 已验证纯文本请求；B=8 是显式 SM75 调优参数，必须保留
安全的因果验证分支。

### Qwen3.8 Flash-Next NVFP4 实验性 PP 路线

这些 profile 面向 `.31` 上的 8 张 Tesla T10（`0,2,3,4,6,7,8,9`），使用
ModelOpt NVFP4 的 SM75 Marlin W4A16 回退、磁盘映射 PLE offload、FP16 KV、无 MTP、
纯文本服务、同步调度、非 eager CUDA Graph，以及 `MAX_BATCHED_TOKENS=512`。以下是固定
4K prompt / 128 completion 测试的中位数，顺序为 prefill / decode tok/s。

| Profile | TP/PP | 后端 | Prefill / decode tok/s | 路线定位 |
|---|---:|---|---:|---|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 2x4 | FlashQLA + PYNCCL | **2379.14 / 24.92** | 推荐均衡路线；默认 `GPU_UTIL=0.92` |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 4x2 | FlashQLA + PYNCCL | **1697.99 / 28.18** | Decode 优先；当前 util 容量 profile |

此前 TP4xPP2 的 `33.93 tok/s` 是特定 shape 的峰值，不应作为稳定承诺；重复
测试通常约为 27--28 tok/s。CAR（custom all-reduce）的 `auto` 模式表示选择
拓扑检测：只有 TP 组全量 NVLink 互联且 P2P 可达时才启用 CAR；只有 PCIe P2P
的 T10 组继续使用 NCCL/PYNCCL。

相同 profile 参数下、按当前发布的 GPU util 默认值测得的 GPU KV token 容量如下。
这些是容量/启动测量，不是额外的吞吐样本。

| Profile | 硬件参考 | GPU util | GPU KV tokens |
|---|---|---:|---:|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 8 张 Tesla T10 | 0.92 | 102,591 |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 8 张 Tesla T10 | 0.96 | 100,031 |
| TP2xPP4 异构参考 | 6 张 Tesla T10 + 2 张 RTX 2080 Ti；TP 组 `[0,2]`、`[8,9]`、`[6,7]`、`[1,5]` | 0.92 | **152,492** |

以上是当前环境的重新实测结果。异构行来自标记为 pre2 的 launcher 日志，
已通过 health 和真实请求；在当前 pre3 树上复测前，不将其升级为 pre3
正式容量承诺。均质行使用 8 张 T10 UUID。TP2xPP4 默认路线先
完成 QSA JIT warmup，再串行执行 10 次 4K/128 请求；平均值为
2469.26 / 24.90 tok/s，表中为中位数。TP2xPP4 在 `GPU_UTIL=0.95` 时可用
115,834 个 token，但 10 次中位数降至 1274.87 / 21.15 tok/s；`0.97` 虽
启动时显示 124,662 个 token，真实请求却触发 FlashQLA 临时 buffer OOM。
TP4xPP2 的 `0.96` 路线在 3 次预热后完成 10 次正式请求且无 OOM；全样本平均值
为 1652.15 / 27.33 tok/s，其中 8/10 个正常 decode 样本的平均/中位数为
28.33 / 28.31 tok/s。

### Qwen3.6 27B AWQ/GPTQ-INT4

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

## 最近一次 INT8KV 4K 参考数据

下面这组短测数据是对上面大上下文条目的补充。本轮按仓库既定的
`PP4096/TG128` 单请求合成口径，重新验证了当前正式保留的 `normal` INT8KV
路线。

| Profile | 测试口径 | 吞吐性能 | 说明 |
|---|---|---:|---|
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | PP4096/TG128 | 1557.20 / 73.79 | 当前正式 FP8 INT8KV `normal` 参考值。 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | PP4096/TG128 | 1684.19 / 69.53 | 当前正式 GPTQ-INT4 INT8KV `normal` 参考值。 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | PP4096/TG128 | 1679.50 / 41.83 | 当前正式 GPTQ-INT4 INT8KV + YaRN `normal` 参考值。 |

这组短测只用于补充上面主表里的长上下文验证数据，不替代原有的长上下文容量证据。
