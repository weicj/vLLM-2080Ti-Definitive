<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向双 RTX 2080 Ti / SM75 推理的终极版 vLLM 运行时。

这是一个硬件定向的 vLLM fork，用来保存已经跑通的 2080 Ti vLLM
栈：补丁源码、启动 profile、运行时说明和稳定环境记录。

核心实测：同一套双 2080 Ti TP=2 runtime 下，Qwen3.6 27B 单请求 decode
达到 `100+ tok/s`，Gemma4 31B 单请求 decode 达到 `~100 tok/s`。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

## 💡 为什么用 RTX 2080 Ti 做 LLM 推理？

2018 年 8 月，NVIDIA 推出了划时代的 RTX 2080 Ti 系列显卡，并将玩家
显卡产品线从 GTX 带入 RTX 时代，从此开启了实时光追时代。这一代显卡
给无数电脑爱好者留下了难以磨灭的印记。八年之后，2080 Ti 依然可以在
2K 分辨率下流畅运行当下主流 3A 大作，可以说是老骥伏枥，志在千里。

而当年的 2080 Ti 还留下了两个非常关键的硬件空间：一是可以把 11 颗
1GB GDDR6 显存颗粒升级为 2GB 容量，从而获得 22GB 可用显存；二是它
保留了在 40 系之后被消费级显卡淘汰的 NVLink 高速互联接口。当高规格
核心、改造后的大显存、高速卡间互联，以及以今天眼光看依然很快的显存
带宽叠加在一起，我们在审视本地 AI 推理时发现，这个组合仍然有巨大的
用武之地。具体而言：

| 指标 | 2x 2080 Ti 22GB + NVLink | 3090 Ti 24GB 基线 | 倍率 |
|---|---:|---:|---:|
| 物理 CUDA core 数量 | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core 数量 | 1,088 | 336 | 3.24x |
| Dense Tensor FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总物理显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存容量 | 44GB | 24GB | 1.83x |
| 二手价格锚点 | CNY 3,600，含 NVLink | 约 CNY 7,000-8,000 | 约 0.5x |

这个项目的核心判断很简单：用约一半 RTX 3090 Ti 二手价格，组出双
22GB RTX 2080 Ti + NVLink，并在 LLM 推理真正关心的物理资源上持平甚至
超过 3090 Ti，再通过 vLLM 运行时优化把这些资源转化成真实 token 产出。

这就是本 fork 的首要价值：把老但仍然很强的 Turing 硅片，通过 Marlin、
FlashQLA/FlashInfer/FA2、TurboQuant/INT8 KV、MTP 和 CUDAGraph 集成，
变成一个严肃可用的 27B/31B 级别推理平台。

## 🧩 核心路线

服务形态：

- 本项目追求的是双 2080 Ti 上的极限单并发性能：一个个人 agent 场景、
  一个足够强的 27B/31B 模型，以及这套硬件能稳定承载的最大实用上下文。
- 它不是多租户 serving 集群。多 agent 使用更适合作为排队式工作区隔离，
  而不是并行长 prefill 吞吐。长上下文并发在调好参数后可以安全排队，
  但在这个 TP=2 profile 下实际会被 runtime scheduler 串行化。

状态：🟢 完整支持；🟡 部分支持；🔴 性能退化；⚪ 不支持。

### Qwen3.6 27B 成熟主线

Qwen 系 27B 是这个 fork 的主要生产路线。它在 Marlin 权重、Native MTP、
FP16/INT8/TQ4NC KV、原生 262K 上下文、YaRN 容量实验和图像多模态兼容性上
覆盖最完整。
快速路径：Qwen 使用 FlashQLA-SM70-SM75 处理 Gated DeltaNet /
linear-attention prefill，full-attention prefill 走 FlashInfer / FA2，
head_dim=256 路线完整保留，decode 侧使用 Native MTP + CUDAGraph 策略。

| 功能 | FP16 KV | INT8 KV | TQ4NC KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 FP8/AWQ/GPTQ | 🟢 FP8/AWQ/GPTQ | 🟢 FP8/AWQ/GPTQ |
| Native MTP3 解码 | 🟢 短上下文速度路线 | 🟢 容量 + 速度路线 | 🟢 压缩容量路线 |
| 原生 262K 上下文 | 🟢 noMTP 真实 prompt 已通过 | 🟡 容量/速度候选 | 🟢 真实 prompt / service 已通过 |
| YaRN 524K 扩展 | ⚪ 非目标路线 | 🟢 容量路线 | 🟡 容量候选 |
| No-eager / CUDAGraph | 🟢 支持 | 🟢 支持 | 🟢 graph-safety 已修复 |
| 快速 prefill 路线 | 🟢 FlashInfer / FA2 | 🟢 FlashInfer / INT8 path | 🟢 TurboQuant FlashInfer path |
| 图像多模态 | 🟢 default-KV 路线 | 🔴 已观察到输出退化 | 🟢 推荐图像路线 |
| Peak MTP3 PP4096/TG128 | 🟢 1747.52 / 100.98 tok/s | 🟢 1744.06 / 81.12 tok/s | 🟢 1746.32 / 85.94 tok/s |

### Gemma4 31B 实验路线

Gemma4 31B 保留为第二路线和实验路线。FP16/default KV 路线速度很好，也有
实用价值；但 Gemma 的 head_dim=512 和异构/GQA attention 让压缩 KV 和长上下文
路线成熟度明显低于 Qwen。
快速路径：Gemma 的 FP16/default KV 短上下文服务可以走快速路线；压缩 KV
长上下文目前仍会回落到 SDPA/GQA 兼容路线，assistant MTP 能兼容但收益更依赖
具体任务。

| 功能 | FP16 KV | INT8 KV | TQ4NC KV |
|---|---|---|---|
| Marlin 权重路线 | 🟢 GPTQ target | 🟢 GPTQ target | 🟢 GPTQ target |
| Assistant MTP 解码 | 🟢 MTP5 峰值路线 | 🔴 实验路径 | 🔴 MTP 性能退化 |
| 原生 262K 上下文 | ⚪ 显存不足 | 🟡 慢速离线路线 | ⚪ 实用容量不足 |
| YaRN 524K 扩展 | ⚪ 不支持 | ⚪ 不支持 | ⚪ 不支持 |
| No-eager / CUDAGraph | 🟢 支持 | 🔴 fallback 很重 | 🟢 兼容性已修复 |
| 快速 prefill 路线 | 🟢 default-KV 快速路径 | 🔴 SDPA/GQA fallback 代价大 | 🟡 短上下文快，长上下文受限 |
| 图像多模态 | 🟢 default-KV 路线 | ⚪ 不支持 | ⚪ 不支持 |
| Peak PP4096/TG128 | 🟢 MTP5 1655.65 / 99.64 tok/s | 🔴 非服务 profile | 🟡 noMTP 1596.15 / 31.70 tok/s |

## 🧪 已测试模型权重

这一节记录 checkpoint 级别的验证结果。这里的标准比“vLLM 能加载”更严格：
支持表示可以启动并生成；推荐表示在双 2080 Ti 上同时具备有意义的速度 /
上下文权衡。

| 模型路线 | 权重量化 | 模型卡 | 状态 |
|---|---|---|---|
| Qwen3.6 27B FP8 | FP8 | [Jackrong/Qwopus3.6-27B-v2-FP8](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) | 🟢 推荐 |
| Qwen3.6 27B AWQ | AWQ-INT4 | [mconcat/Qwopus3.6-27B-v2-AWQ-4bit](https://huggingface.co/mconcat/Qwopus3.6-27B-v2-AWQ-4bit)<br>[QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) | 🟢 推荐 |
| Qwen3.6 27B GPTQ | GPTQ-INT4 | [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) | 🟢 推荐 |
| Qwen3.6 27B AutoRound | AutoRound INT8 | [Minachist/Qwen3.6-27B-INT8-AutoRound W8A16-GS128](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound/tree/W8A16-GS128) | 🟡 支持 |
| Gemma4 31B GPTQ | GPTQ-INT4 + assistant draft | [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | 🟡 支持 |

FP8 和 AutoRound INT8 结论：FP8 是追求质量时推荐的高质量 8bit Qwen 路线。
它在 SM75 上走的是 weight-only FP8，不是原生 FP8 compute，所以 4-bit
AWQ/GPTQ 仍然是默认性能 / 容量路线。AutoRound GS128 的 KV 容量太低，main
分支容量更好但 speculative decode 速度损失过大，所以测试模型列表只保留
GS128。
Qwen3.6 原版 AWQ 仍然是有价值的 baseline，后续回归测试可以重新下载；
但质量路线已经由 Qwopus 取代。

## 🛠️ 目标硬件与运行环境

- 已验证 GPU profile：双 RTX 2080 Ti 22GB，SM75，NVLink，tensor parallel
  size 2
- CUDA/PyTorch：CUDA 12.8，`torch 2.11.0+cu128`
- 基础 vLLM：`0.21.0`
- 仓库身份：`vllm-2080ti-definitive`
- 稳定运行时身份：`vllm-sm75-tp2-cu128`
- 兼容目标：NVIDIA Turing / SM75 显卡。其它 Turing 显卡仍需要按显存容量、
  P2P/NVLink 行为、模型 head_dim、KV dtype、CUDAGraph/MTP 设置重新验证
  profile。

## 🚀 MTP 与 KV 精度

MTP / speculative decoding 不是固定倍率加速。它的收益取决于 target model
最终接受了多少 draft token，所以最佳 `MTP_K` 会随任务类型变化。我们的 sweep
里，代码生成和科学计算分析的加速更明显；文学/自然文长输出的接受率更容易下降，
高 K 反而可能在真实长输出里回退。

KV 精度会从另一个方向影响同一条 decode 路线。开启 MTP 时，FP16/default KV
保持最佳长上下文 decode 速度；未开启 MTP 时，当前 Qwen3.6 sweep 中 FP16 和
INT8 KV 没有体现出明显 decode 速度下降。TurboQuant KV 在短上下文测速中仍然
很快，但长上下文下会明显掉速，因此更适合在优先追求更大 KV 容量时使用，而不是
作为最大长上下文 decode 速度路线。

当前稳定 profile 中，Qwen3.6 以 MTP3 作为更保守的混合 agent 默认值；Gemma4
只在特定 FP16 速度 profile 里使用更高 MTP。详细实测表见
[MTP 任务敏感性](docs/mtp-task-sensitivity.md)；FP8 和 GPTQ-INT4 下不同 KV
精度的 A/B 数据见
[Qwen3.6 KV 吞吐 Sweep](docs/qwen36-kv-throughput-sweep.zh-CN.md)。

## 🧭 Profile 与推荐路线

生产 profile 选择单独维护在
[docs/model-profile-routes.zh-CN.md](docs/model-profile-routes.zh-CN.md)。
这里统一说明路线限制、Qwen/Gemma 推荐 profile、KV 精度选择、上下文窗口目标、
MTP 设置、多模态限制、并发假设，以及 `bench_tools/stable_profile.sh` 使用的
launcher alias。

## 🚀 如何使用

这个 fork 更适合作为目标 SM75 主机上的固定 runtime tree 使用，而不是
直接当成通用 `pip install vllm` 替代品。

1. 把 patched runtime 放到目标机器上的稳定目录。路径按自己的机器设置：

   ```bash
   export STABLE_ROOT=/path/to/vllm-sm75-tp2-cu128
   ```

2. 把模型权重放到稳定模型目录。启动 profile 默认会寻找这些别名路径：

   ```text
   $MODEL_ROOT/qwen-family-27b-awq
   $MODEL_ROOT/qwen-family-27b-gptq-int4
   $MODEL_ROOT/qwen-family-27b-fp8
   $MODEL_ROOT/gemma4-family-31b-gptq
   $MODEL_ROOT/gemma4-family-assistant
   ```

   如果你使用其它 checkpoint，也可以直接覆盖 `MODEL_DIR`。

3. 启动一个已验证 profile：

   ```bash
   cd "$STABLE_ROOT"

   PROFILE=qwen27-awq-mtp3-peak \
   MODEL_ROOT=/path/to/models \
   PORT=8000 \
   CUDA_VISIBLE_DEVICES=1,2 \
   RUN_USER=vllm \
   RUN_HOME=/var/lib/vllm \
   ./bench_tools/stable_profile.sh
   ```

4. 检查 OpenAI-compatible endpoint：

   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

## ❓ 硬件 Q&A

**Q：需要什么样的卡间互联？**

A：推荐 NVLink，但真正的底线是 GPU 之间能开启 PCIe P2P。当前验证系统使用了
NVLink，而且 PCIe 拓扑本身很不理想：一张卡 PCIe 3.0 x1，另一张卡 PCIe 3.0
x4。在 NVLink 承担 GPU-to-GPU 通信时，PCIe 插槽带宽不是主要瓶颈。没有
NVLink 时，不能直接认为极窄 PCIe 带宽也足够，仍然需要确认 P2P 行为并按实际
拓扑 benchmark。

**Q：需要很强的 CPU 或很多内存吗？**

A：不需要。已验证路线可以跑在低端桌面 CPU + 16GB RAM 这类较低规格的平台上。
更强 CPU/更大内存主要帮助 compile cache、下载和本地 build，不是
steady-state token generation 的核心瓶颈。

**Q：哪些 Turing 显卡值得尝试？可以 11GB + 22GB 混搭吗？**

A：完整验证目标是双 RTX 2080 Ti 22GB。其它更推荐高显存 TU102 级别显卡：
TITAN RTX 24GB、Quadro RTX 6000 24GB、Quadro RTX 8000 48GB，最好成对使用并
具备 NVLink 或确认可用的 PCIe P2P。不推荐 11GB + 22GB RTX 2080 Ti 混搭来跑
这些 27B/31B profile，因为 vLLM TP=2 基本会被较小 rank 的显存限制。更小的
Turing 卡可以跑小模型，但不是这个 stack 的主要目标。

**Q：已验证的 CUDA、PyTorch 和驱动版本是什么？**

A：stable runtime 是 CUDA 12.8 + `torch 2.11.0+cu128`。请使用支持目标
GPU、并且兼容该 CUDA runtime 的较新 NVIDIA driver。不要随意混用
build/runtime 假设：PyTorch CUDA 版本、本地 CUDA toolkit、FlashInfer/FlashQLA
构建和启动 profile 应保持一致。

**Q：还有哪些硬件风险需要注意？**

A：散热、供电稳定性，以及给模型文件和 compile cache 留够 SSD 空间。长 prefill
或反复 CUDAGraph/AOT 编译时，降频很容易伪装成软件性能回退。

## 🔗 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双
  2080 Ti 模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。
  本仓库则聚焦于 vLLM 运行时源码、补丁和启动配置本身。

## 🙏 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向
fork，遵循 Apache-2.0 license。仓库保留上游项目结构，并加入面向双
2080 Ti / SM75 路线的本地运行时补丁、启动 profile 和验证记录。

稳定运行时使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving
  框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：vLLM 使用的
  attention、sampling 和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  面向 SM70/SM75 的适配版本，稳定 Qwen3.6 prefill profile 会用到。
- FlashAttention / FA2、TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM
  相关加速 kernel：这些都是已有开源加速工作，本项目将它们整合、适配并在
  目标硬件上验证。
