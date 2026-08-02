# Huihui-Qwen3.6-27B-abliterated-AWQ-MTP 本地测试报告

测试日期：2026-08-02（Asia/Shanghai）

## 1. 测试对象与边界

- 模型：`shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`
- 本地路径：`/home/ubuntu/storage/llm/Qwen/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`
- 权重检查：10 个主模型 safetensors 分片和 `model_extra_tensors.safetensors` 均按 index 精确匹配大小，并成功通过 `safe_open` 校验。
- 测试 GPU：CUDA_VISIBLE_DEVICES=6,7，TP=2。
- GPU 4/5 是生产卡；测试前后均保持约 4 MiB，占用没有被测试进程触碰。
- 代码：weicj fork `vllm-2080ti-deifinitive`，commit `0a0caa1098e941b0314443219e8b9c9dd25a14a4`。

## 2. 实际运行配置

- AWQ Marlin W4A16，dtype=FP16。
- MTP=3，目标模型和 draft 使用同一 MTP 权重；开启图像输入，`image=1, video=0, audio=0`。
- `max_model_len=245760`，`max_num_batched_tokens=2048`，`max_num_seqs=1`，GPU utilization=0.92。
- 目标 attention 和 draft attention 都显式指定 `TRITON_ATTN`；GDN/Mamba prefill 使用 Triton。
- CUDA Graph 保留：`cudagraph_mode=PIECEWISE`，capture size=4，`enforce_eager=0`。
- custom all-reduce 对照两组均完成：`DISABLE_CUSTOM_ALL_REDUCE=1` 与 `=0`。
- 为适配当前机器的 CUDA/编译环境，使用 `NCCL_NET=Socket`、`NCCL_IB_DISABLE=1`、`NCCL_NET_PLUGIN=none`，主机编译器使用 GCC 11，并关闭 FlashInfer sampler JIT，改用 PyTorch-native sampler；这不等于关闭 FlashInfer attention。

## 3. 结果

| 配置 | 文本 256 token 中位端到端速度 | 图像 128 token 中位端到端速度 |
|---|---:|---:|
| MTP3 + custom all-reduce 关闭 | 48.766 token/s | 30.435 token/s |
| MTP3 + custom all-reduce 开启 | 47.670 token/s | 29.521 token/s |
| 开启相对关闭 | -2.25% | -3.00% |

文本测试为 2 次 warmup + 5 次测量；图像测试为 1 次 warmup + 3 次测量，`temperature=0`、`enable_thinking=false`。两组文本和图像请求均返回 HTTP 200，图像能够正常识别并生成中文描述。

custom all-reduce 开启时日志出现两侧各 `Registering 130 cuda graph addresses`，同时仍为 `CUDAGraphMode.PIECEWISE`；没有发生 CUDA Graph 退化为 eager，也没有出现共享内存广播卡死。

开启统计日志后，MTP 的实际接受率不是固定值：连续文本窗口的平均 draft acceptance rate 约 67.8%–70.6%，第三位置接受率约 0.49–0.52；图像请求窗口约 47.2%，三位置接受率约 0.679/0.415/0.321。也就是说 MTP 确实在工作，但并不是每轮都能完整接受 3 个 draft token。

## 4. 启动问题与最终处理

1. NCCL 默认网络插件在当前混合 CUDA/驱动环境中触发 `ncclNetPluginInit` 崩溃；强制 Socket、禁用 IB 和外部 net plugin 后稳定。
2. launcher 原先硬编码 `/usr/bin/gcc-12`，机器缺少对应 `cc1plus`；切换 GCC 11 后通过。
3. CUDA 12.9 下 FlashInfer sampler 的 bfloat16 重载出现编译歧义；仅关闭 FlashInfer sampler，保留 Triton attention 和 CUDA Graph。
4. 245760 上下文在 GPU utilization=0.90 时 KV 余量不足；提高到 0.92 后通过 smoke test。
5. draft 默认选择 FlashInfer prefill 时，head_dim=256 触发 invalid argument；为目标和 draft 同时显式指定 `TRITON_ATTN` 后通过。

这些是当前运行时、编译器和 SM75 组合的兼容性问题，不是该模型权重损坏。

## 5. 文件索引

- custom all-reduce 关闭文本结果：`huihui-text-mtp3-aroff.json`
- custom all-reduce 开启文本结果：`huihui-text-mtp3-aron.json`
- custom all-reduce 关闭图像结果：`huihui-image-mtp3-aroff.json`
- custom all-reduce 开启图像结果：`huihui-image-mtp3-aron.json`
- 稳定启动日志：`vllm-qwen27b-int4-fp16kv-240K-mtp3-text-image-cu128-20260802-025021.log`

## 6. 结论

这个模型适合继续做本地多模态性能测试，AWQ、MTP、图像输入、TP2 和 CUDA Graph 均已实际跑通。就这台 RTX 2080 Ti 双卡、当前 weicj runtime 和单并发端到端请求而言，custom all-reduce 没有带来收益，反而慢约 2%–3%，因此暂不建议生产默认开启；它可以作为可选实验开关保留。当前主要瓶颈仍是 SM75 上 AWQ 解量化、Mamba/GDN 路径、TP 跨卡通信、draft 接受率以及单序列端到端调度，而不是“27B 但每次只激活 3B”这一项参数本身。

## 10. GPTQ 参考权重与全图模式的最终 A/B（2026-08-02）

为了把“checkpoint/量化格式”和“runtime/主机”拆开，另外下载并校验了项目 README
列出的 `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`。
该目录的 5 个 GPTQ shard、`model-auxiliary.safetensors`、index 和 tokenizer 都通过
`safetensors.safe_open` 校验，随后在同一对 GPU 6/7、同一个代码 commit、同一个
4096/128 `profile_request.py` 口径下测试：

| A/B 配置 | 结果 | 说明 |
|---|---:|---|
| 当前 Huihui AWQ + PIECEWISE + safe sync | 约 73.6–74.6 tok/s | JIT 预热后的稳定区间 |
| 项目 GPTQ + `gptq_marlin` + PIECEWISE + safe sync | 约 71.0–72.2 tok/s | 不是量化格式单独造成 94.48 的差距 |
| 项目 GPTQ + PIECEWISE + `VLLM_SM75_SPEC_SYNC_MODE=nosync` | 约 64.8–65.6 tok/s | 对本机反而变慢，不建议生产使用 |
| 项目 GPTQ + FULL_AND_PIECEWISE + GPU util 0.95 | 约 50.6–52.8 tok/s | 输出重复/异常，且稳定性不合格 |

FULL graph 在 GPU util=0.92 时还会直接因为额外的约 0.34 GiB graph workspace
导致 KV cache 不足而无法启动；提高到 0.95 虽然能启动，但 Mamba/GDN speculative
状态被错误复用，纯 filler 也输出了 `1989` 等非预期内容。因此“保留 CUDA Graph”
并不等于必须强行开启 Mamba 的 speculative FULL graph；当前可用的正确路线是
`PIECEWISE + safe sync`。

这组对照排除了几个常见误判：GPTQ Marlin 并没有自动得到项目表里的 94.48，
`nosync` 不是免费加速开关，FULL graph 也不是当前 SM75 混合 Mamba/GDN 模型的
稳定快路径。当前 AWQ 的 74–75 tok/s 是真实可复现的最佳稳定值。

## 11. 为什么项目表能到 94，而这台主机到不了

这里的“同样配置”只覆盖了模型参数和 vLLM 参数，并不代表运行条件相同。项目自身
的 README 已记录：同一条 4096/128 GPTQ-INT4 MTP3 路线，在 `i3-9100T` 上约
91 tok/s，而在双 Xeon X5675 上约 56 tok/s。当前机器是双路 Xeon Platinum 8153，
单核标称 2.0 GHz；vLLM 启动日志还明确显示把 Torch CPU parallelism 从 32 降为 1。
这类单请求 speculative 调度、CPU/GPU event 和 TP 控制面延迟会直接进入 decode
时间，因此不能只看 GPU 型号推断应有的 tok/s。当前 74/94≈79%，与“低频双路
服务器 CPU 对高频单路桌面 CPU 的控制面差距”是相符的强证据；NUMA 绑定本身不是
主要因素，但 CPU 单核频率/调度延迟仍然是剩余差距的重要候选。

另外，项目验证栈写的是 CUDA 12.8 + `torch 2.11.0+cu128` + FlashInfer
0.6.8 + driver 590.48.01；本机 Python 进程实际加载的是 torch `2.11.0+cu130`，
驱动 580.173.02，本地 `nvcc` 12.1。驱动升级到“支持 CUDA 12.9/13.0”并不等于
把 Python runtime 变成项目验证的 cu128。CUDA/torch/FlashInfer 组合会改变 Triton
和 Marlin 的编译缓存及 kernel 选择，这也是另一项确定的非同条件。

所以目前的结论不是“模型只配跑到 74”，而是：在这台双路 2.0 GHz Xeon、cu130
runtime、当前 Huihui AutoRound-AWQ checkpoint 上，74–75 是已经补齐 FlashQLA、
FlashInfer sampler、MTP3 和 PIECEWISE CUDA Graph 后的稳定结果；README 的 94.48
是另一套 checkpoint、验证软件栈和主机控制面条件的数字。若要继续逼近 94，优先级
应是用项目 cu128 环境在高单核频率主机上重跑项目 checkpoint，再对照当前 AWQ；
继续改 custom all-reduce、MTP_K 或强开 FULL graph 不会带来可靠的 1.3 倍提升。

## 7. 与项目公开数字的追加对照（2026-08-02）

为排除“测试方法不同”的影响，按项目 `tools/profile_request.py` 的口径重新测试：`completions`、纯 filler、4096 prompt tokens、128 generated tokens、`ignore_eos`、1 次 warmup + 3 次测量。结果如下：

| 对照 | decode 中位速度 |
|---|---:|
| 当前环境 + MTP3 + Triton attention | 52.66 token/s |
| 当前环境 + MTP=0 + Triton attention | 38.86 token/s |
| 当前环境 + MTP3 + Triton，绑定 GPU 所在 NUMA CPU | 52.49 token/s |
| 仅 overlay 项目锁定的 FlashInfer 0.6.8，MTP3 | 61.26 token/s |
| 项目 README 中 Qwen3.6-27B AWQ 文本+图像参考 | 94.48 token/s |

因此 MTP 本身没有失效：`38.86 -> 52.66`，提升约 1.36 倍；纯 filler 测试中日志显示三层 draft 的接受率为 100%。绑定 NUMA 后几乎不变，说明这次差距不是简单的 CPU 亲和性问题。FlashInfer 0.6.8 的独立 overlay 将 52.66 提升到 61.26，证明 attention 后端是重要差异，但仍不足以解释全部差距。

## 8. 性能差距的根因定位

1. **实际运行时不是项目验证栈。** 项目声明的验证组合是 CUDA 12.8、torch `2.11.0+cu128`、FlashInfer `0.6.8.post1`、参考驱动 `590.48.01`；本机实际进程是 torch `2.11.0+cu130`、CUDA runtime 13.0、FlashInfer `0.6.11.post2`，本地 toolkit 则落到 CUDA 12.1。launcher 中的 “Validated CUDA/Torch: CUDA 12.8 / torch cu128” 是项目元数据，并不代表 Python 进程真的加载了 cu128。该混用会改变 kernel 选择、编译结果和 FlashInfer/采样器兼容性。
2. **当前稳定配置被迫绕开了项目的快路径。** 由于当前组合下 FlashInfer sampler 的 `bfloat16` 编译歧义，以及默认 FlashInfer head_dim=256 的 invalid argument，稳定运行配置使用了 PyTorch-native sampler 和 `TRITON_ATTN`。日志明确记录 `VLLM_USE_FLASHINFER_SAMPLER=0`、目标/draft 为 Triton；FlashInfer 0.6.8 overlay 恢复 attention 后速度提高约 16%，但 sampler 仍无法在现有 CUDA 头文件组合下编译。因此“同样启动参数”实际执行的 kernel 路径并不相同。
3. **公开数字不是该 checkpoint 的保证值。** 项目表格测试的是其指定的 Qwen3.6-27B AWQ/GPTQ checkpoint；当前 `shawnw3i/Huihui...` 的量化配置是 AutoRound AWQ（group 128、zero-point），且包含自身的 MTP/附加张量。量化 scale、权重布局、MTP 校准和版本都可能不同。它们架构名称相同，不等于 Marlin 生成的 kernel 工作量和 MTP 接受率相同。
4. **比较指标曾经不一致。** 项目表中的第二个数字是固定 4096/128 synthetic benchmark 的 decode tok/s，不是普通 API 请求从首 token 到结束的端到端平均值。此前 48 token/s 的结果不能直接和 94.48 比；现在已经按项目口径复测，仍有差距，但差距被准确限定为约 `94.48/52.66=1.79x`，而不是原先看起来的近 2 倍以上。
5. **custom all-reduce 和 CUDA Graph 不是主因。** CUDA Graph 实际处于 `PIECEWISE`，capture size=4；关闭/开启 custom all-reduce 的端到端结果分别约 48.77/47.67（文本）和 30.44/29.52（图像），开启反而慢 2%–3%。它只影响 TP collective，不能补回 attention、sampler 和 runtime 版本造成的差距。

## 9. SM75 完整快路径复测（FlashQLA + FlashInfer sampler）

之后继续把项目中的 SM75 FlashQLA legacy GDN 路径补齐，并在 CUDA 12.1
toolkit / 当前 CUDA 13.0 torch 进程下使用隔离的 FlashInfer `0.6.8.post1`
overlay。为避免 PyTorch JIT 在当前 GCC/CUDA 组合中重新编译失败，FlashQLA
扩展使用本地预编译的 SM75 `.so`；这不会影响 4/5 生产卡。

稳定测试配置保持：AWQ Marlin、FP16 KV、多模态 image=1、MTP3、CUDA Graph
`PIECEWISE` capture=4、`max_num_batched_tokens=2048`、custom all-reduce
关闭。结果为：

| 配置 | 4096/128 decode tok/s |
|---|---:|
| FlashQLA + FlashInfer 0.6.8 attention + native sampler | 约 61–65 |
| FlashQLA + FlashInfer 0.6.8 attention + FlashInfer sampler | **74.3–74.9** |
| 上述配置，`max_num_batched_tokens=4096` | 约 68–73 |
| 上述配置，MTP2 | 约 54–56（稳定两次）|

因此当前 checkpoint 的最佳稳定值是约 **74–75 token/s**；MTP2、4096
batched tokens 和 GDN spec warp=8 的局部微基准优势都没有转化为端到端收益。
CUDA Graph 在这些测试中始终保留，并与 FlashQLA、FlashInfer sampler 同时工作。

同一配置随后进行了真实图片 smoke：`chat/completions` 返回 HTTP 200、64 个
输出 token 正常结束，模型能识别图片中的蓝色方块等内容；该请求的首 token
时间约 4.83 秒，后续解码约 15.3 token/s（图片编码/视觉 prefill 不应与纯文本
4096/128 decode 数字混为一谈）。服务日志同时给出该短请求的 MTP 平均接受率
约 78.9%，证明图像链路也确实走了 MTP，而不是退回 noMTP/eager。

项目 profile 表中的 `1760.14 / 94.48` 对应的是项目列出的
`QuantTrio/Qwen3.6-27B-AWQ`、`mconcat/Qwopus...AWQ` 或
`llmfan46/...GPTQ` 测试 checkpoint；当前 `shawnw3i/Huihui...` 不在该表中，
其 AutoRound AWQ 配置还明确保留了大量线性注意力和视觉模块为未量化 FP16。
所以即使架构名、MTP=3、AWQ 和启动参数相同，也不能把 94.48 当成当前
checkpoint 的硬件上限。当前 74–75 已经把此前未启用的 SM75 FlashQLA 和
FlashInfer sampler 快路径补上，剩余差距主要属于 checkpoint/量化布局和
实际运行时（cu130 vs 项目 cu128）差异，而不是 CUDA Graph 或 MTP 没生效。

结论：目前最主要的断点在“验证栈不一致 + FlashInfer/FlashQLA 快路径没有完整启用”，其次是 checkpoint/量化版本与项目基准不同；MTP、CUDA Graph、custom all-reduce、GPU 6/7 的 NUMA 亲和性都已通过对照排除为第一嫌疑。要有意义地追 README 的 94 token/s，下一步应先复原完整 CUDA 12.8/cu128/FlashInfer 0.6.8 隔离环境，再用项目基准 checkpoint 做 A/B；仅继续调 `MTP_K` 或 custom all-reduce 不会得到 1.8 倍收益。

这个模型适合继续做本地多模态性能测试，AWQ、MTP、图像输入、TP2 和 CUDA Graph 均已实际跑通。就这台 RTX 2080 Ti 双卡、当前 weicj runtime 和单并发端到端请求而言，custom all-reduce 没有带来收益，反而慢约 2%–3%，因此暂不建议生产默认开启；它可以作为可选实验开关保留。当前主要瓶颈仍是 SM75 上 AWQ 解量化、Mamba/GDN 路径、TP 跨卡通信、draft 接受率以及单序列端到端调度，而不是“27B 但每次只激活 3B”这一项参数本身。
