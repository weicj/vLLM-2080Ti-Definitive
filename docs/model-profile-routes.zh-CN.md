# 模型 Profile 路线

本文记录部署 profile 的证据口径。当前 profile 清单、含义和实测吞吐统一维护在
[Profile 导引](../profiles/README.zh-CN.md)。

`Profile` 是 `launcher.sh` 选择的相对 `.env` 路径；具体 checkpoint 仍然通过
`MODEL_DIR` 单独选择。

## 证据口径

完整通过表示真实请求返回 HTTP 200、stream 正常结束，并且中文质量 smoke 没有
重复、残缺、乱码或明显答非所问。

平台期通过只说明容量风险较低；如果质量 smoke 失败，即使容量或合成吞吐可用，
也不晋升 profile。

只 load 成功、READY、health 通过、小窗口 smoke、空 stream，都不算容量证据。

## KV 精度定位

- FP16/default KV 是质量路线。
- INT8 KV 是容量 / 平衡路线；当前只保留 `normal` / piecewise profile。
- TQK8V4 是 TurboQuant 压缩路线；除质量通过的 `fast` profile 外，也用于
  Qwen3.8 NVFP4 DFlash2 的 `normal` 路线；该路线使用因果安全的投机 target
  验证路径。
- TQ4NC 有过容量实验，但当前正式 profile 不采用。

## 说明

- Profile 按 `profiles/<model>/<mode>/<weight>/<route>.env` 组织。
- `normal` 是当前推荐生产路线；`fast` 只保留质量 smoke 通过的高性能路线；
  `safe` 是 launcher 的 eager 回退档，不作为当前正式 profile 目录。
- 同一套双 2080 Ti runtime 也已经验证了 Qwen3.6 35B FP8 MoE 路线。正式
  预设现在覆盖 256K `normal` / `aggressive` noMTP 纯文本路线、136K
  `normal` / `aggressive` noMTP 图文路线，以及一条 178K `fast` MTP3
  速度预设。
- FP8 + FP16KV `normal` 正式上下文为 256K，并且 `262016/128` 长提示 smoke
  已通过。
- FP8 + FP16KV `aggressive` 也已验证到 256K。正式记录的吞吐仍以
  `4096/128` 合成短测为准；接近满长的 `262016/128` 只作为容量 smoke，
  因为长跑里流式 chunk 合并会把 decode 速度抬高。
- FP8 + FP16KV 图文路线在 `normal` 和 `aggressive` 下都已验证到 136K。
  两条都通过了 `138240/128`，边界上的 `139008/64` 也可通过，而
  `139136/32` 会超过配置的 `139264` 上限。
- FP8 + FP16KV `fast` 当前验证到 178K，并且 `182144/128` 长提示 smoke 已通过。
- FP8 + TQK8V4 已验证 256K 纯文本和 240K 图文；图文路线使用 GPU util 0.96。
- Qwen3.8 27B NVFP4 + DFlash2 是首条晋升的 DFlash2 路线，使用
  `nvidia/Qwen3.8-27B-NVFP4`、`incoai/Qwen3.8-27B-DFlash2`、TQK8V4、K=7、
  草稿侧 `TRITON_ATTN` 和 PIECEWISE CUDA Graph。SM75 的 target 验证路径保留
  因果 sequence length，并以 B=8 分块。高接受率固定 4K/128 三次 decode
  分别为 155.16、166.57、166.76 tok/s，平均 162.83。该路线已通过 4K 正确性
  请求、接近满长的 262016-token prompt smoke，以及真实 4K/8K token
  HTML/JavaScript 长输出；没有 stream 失败、CUDA illegal instruction 或
  EngineDead。它是纯文本路线，正式上下文为 256K。
- fast + INT8KV 不保留：容量或合成速度可以成立，但中文质量 smoke 出现重复或
  残缺输出。
- Qwen3.8 Flash-Next NVFP4 识别为 `qwen4` 架构，并作为 8 卡 T10 的实验性
  PP 路线保留。发布 profile 使用已验证的 `PLE_PLACEMENT=disk`；`cpu` 与
  `gpu` 保持显式 opt-in。当前环境下，
  推荐的 TP2xPP4 profile 在 `GPU_UTIL=0.92` 时实测 102,591 个 GPU KV tokens。
  完成 QSA JIT warmup 后串行执行 10 次固定 4K/128，中位数为
  2379.14 / 24.92 tok/s，平均值为 2469.26 / 24.90 tok/s。TP4xPP2 在
  `GPU_UTIL=0.96` 时实测 100,031 个 GPU KV tokens，中位数为
  1697.99 / 28.18 tok/s，平均值为 1652.15 / 27.33 tok/s；其中 8/10 个
  decode 样本稳定在约 28.3 tok/s。历史上的 33.93 tok/s 属于特定 shape 峰值，
  不是稳定承诺。
- profile 的 CAR `auto` 模式表示选择拓扑检测。仅在 TP 组每张卡同时满足 P2P
  可达且全量 NVLink（ROCm 为 XGMI）互联时自动启用。只有 PCIe P2P 的 T10 组
  继续使用 NCCL/PYNCCL，这是已验证的
  decode 路线。
- 当前环境的容量证据为：均质 8 张 T10 的 TP2xPP4 在 `GPU_UTIL=0.92` 时可用
  102,591 个 GPU KV tokens；均质 8 张 T10 的 TP4xPP2 在 `GPU_UTIL=0.96`
  时可用 100,031 个。另有一组异构 TP2xPP4（6 张 T10 + 2 张 RTX 2080 Ti，
  TP 组 `[0,2]`、`[8,9]`、`[6,7]`、`[1,5]`）在 `GPU_UTIL=0.92` 启动得到
  **152,492** 个 token，并通过 health/真实请求；其 launcher 日志标记为
  pre2，需在当前 pre3 树复测后才能作为 pre3 容量晋级。TP2xPP4 在
  `0.95` 时可用 115,834 个 token，但 10 次 4K/128 中位数降至
  1274.87 / 21.15 tok/s；`0.97` 虽启动时有 124,662 个 token，真实请求
  却在 FlashQLA 临时分配处 OOM。
- 旧 `fast` + INT8KV 的兼容问题仍应修复，但这不代表把该路线重新晋升为
  fast 正式 profile。
- 吞吐背景记录见
  [Qwen3.6 KV 吞吐 Sweep](qwen36-kv-throughput-sweep.zh-CN.md) 和
  [MTP 任务敏感性](mtp-task-sensitivity.md)。
