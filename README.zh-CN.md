<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-cover.jpg)

面向双 RTX 2080 Ti 22 GB / SM75 推理的硬件定向 vLLM fork。本分支是 `vllm-2080ti-definitive-0.2.x` 维护线：基于上游 vLLM `v0.27.1`、CUDA 13.0 和 PyTorch 2.13 的公开 `0.2.x` 预发布线。它不是稳定生产默认版本；持续维护的 `0.1.x` 线仍是项目当前的稳定发布主线，对应 CUDA 12.8 与 PyTorch 2.11。

项目保留复现双 2080 Ti TP=2 栈所需的 SM75 专用源码修改、launcher profile 和
测试证据。它基于上游 vLLM；再发布派生版本时必须保留上游许可证、上游署名以及
`github.com/weicj` 的项目署名。

语言：[English](README.md) | 简体中文

![单请求实时测速演示](docs/assets/vllmspeed.gif)

Fork 版本：`0.2.1-pre3`
基础 vLLM：`0.27.1`

分支：[`vllm-2080ti-definitive-0.2.x`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/vllm-2080ti-definitive-0.2.x)
预发布快照：[v0.2.1-pre3](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.1-pre3)
相比 pre2 的变化：[CHANGELOG.md](CHANGELOG.md)

## 为什么用 RTX 2080 Ti 做 LLM 推理？

这个项目的判断很实际：两张通过 NVLink 连接的 22 GB RTX 2080 Ti 提供 44 GB
显存、较高的显存带宽和 136 个 Turing SM。经过针对 SM75 的 vLLM 适配后，这套
硬件不只是运行小模型，也足以承载严肃的本地 27B 与 35B 级模型服务。

| 指标 | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB 基线 | 倍率 |
| --- | ---: | ---: | ---: |
| 物理 CUDA core | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| 物理 Tensor Core | 1,088 | 336 | 3.24x |
| Dense FP16 矩阵吞吐 | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存 | 44 GB | 24 GB | 1.83x |

本 fork 通过 Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP 和 CUDA
Graph，把这些硬件资源转成可用的 serving 栈。

## 当前状态

`0.2.x` 的目标环境是 Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15、CUDA 13.0 与 PyTorch 2.13。持续维护的 `0.1.x` 线仍是 CUDA 12.8、PyTorch 2.11、较早 kernel 以及 GCC 12/13/14 的兼容路线。

双 2080 Ti 的 CUDA Graph 验证记录在
[迁移验证报告](docs/2080ti-0.2.1-pre-validation.md)：其中包含实际显卡选择规则、
构建门槛、正确性回归以及 4K/128 测试口径。仅能加载的 checkpoint 不应被视为
已提升为部署路线。

此前已合并 SM75 PR 的迁移判断见 [0.2.x PR 迁移审计](docs/0.2.x-pr-migration-audit.md)。

Launcher 支持选择 tensor parallel（`TP_SIZE`）和 pipeline parallel（`PP_SIZE`），
当可见 GPU 数量与拓扑要求匹配时可以启动 TP/PP 混合推理。当前主要验证部署仍是双
RTX 2080 Ti、TP=2、PP=1；其他并行布局可用于工程测试，但需要单独完成验证。

## 已测试模型权重

当前模型和权重路线。具体服务预设和性能数据见
[Profile 导引](profiles/README.zh-CN.md)。

| 模型路线 | 权重路线 | 模型卡 | 推荐场景 |
| --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 高精度单并发 |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 长上下文多并发 |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | 个人快速推理 |

## 构建与启动

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
git switch --track origin/vllm-2080ti-definitive-0.2.x
./build.sh
```

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=qwen27b/w8a16/fast/tqk8v4-256K-mtp3-text-only.env \
MODE=fast GPU_DEVICES=4,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

使用 `./launcher.sh` 进入交互式配置，或使用 `./launcher.sh --print-config` 预览路线。
可用 profile 见 [Profile 导引](profiles/README.zh-CN.md)。

## Profile 与推荐路线

从 [Profile 导引](profiles/README.zh-CN.md) 开始选。Profile 按
`profiles/<model>/<weight>/<mode>/<route>.env` 组织，例如
`qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env`、
`qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env` 和
`qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env`。

可用模式：

- `normal`：稳定的日常部署模式。
- `fast`：更高性能模式，只用于已验证路线。
- `aggressive`：性能最高但质量风险也最高。
- `safe`：用于排障和兼容性的保守回退模式。

Profile 只选择路线参数。GPU、端口、模型路径、chat template 和 reasoning 默认值
由 launcher 统一管理；实验性 profile 也可以携带已验证的 TP/PP 布局。

## MTP 与 KV 精度

优先使用项目自带 profile，不要一开始手动调 MTP 和 KV。KV 先按目标选择：
FP16/default KV 追求输出质量，INT8 KV 用于平衡型长上下文服务，TurboQuant K8V4
用于压缩 fast 路线。MTP 收益取决于接受率，合成峰值必须再用真实输出和质量探针
检查。

当前迁移请以[验证报告](docs/2080ti-0.2.1-pre-validation.md)中的精确方法和数据
为准，尤其是 TurboQuant 与 MTP3。历史 profile 容量不是 cu130 证据。
INT6/AutoRound checkpoint 需要 `humming-kernels[cu13]==0.1.13`，这是本分支锁定的版本；
完整 Minachist 路线仍未验证。

## 目标硬件

- 两张经 NVLink 连接的 RTX 2080 Ti 22 GB
- NVIDIA Turing / SM75，tensor parallel size 2
- `0.2.1-pre3` 目标：CUDA 13.0、PyTorch 2.13、Python 3.12
- 目标主机：Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15

其它 Turing 显卡仍需针对显存容量、PCIe/NVLink 拓扑、模型 head dimension、
KV cache dtype 和 CUDA Graph 行为独立验证。

## 硬件 Q&A

**需要什么样的卡间互联？**

推荐 NVLink。PCIe P2P 是底线，但没有 NVLink 时不能把窄 PCIe 链路直接视为已验证
替代方案；应先确认 P2P，再按实际主机拓扑测试。

**需要很强的 CPU 或很多内存吗？**

不需要高端 CPU，但现代单核性能和较低的平台延迟很重要。更多内存主要帮助构建、
下载和 compile cache；即使 GPU 相同，老旧 CPU 平台也可能降低 decode 吞吐。

**可以混用 11 GB 和 22 GB Turing 卡吗？**

不建议用于文档中的 27B/35B TP=2 路线。TP 会受到较小 rank 显存的限制。更好的
候选是成对的高显存 TU102 卡，例如 TITAN RTX、Quadro RTX 6000 或 Quadro RTX
8000，并且要有 NVLink 或确认可用的 PCIe P2P，之后仍需独立验证 profile。

**应该使用哪些 CUDA 和 PyTorch 版本？**

`0.2.1-pre3` 目标是 CUDA 13.0 + PyTorch 2.13。旧的 CUDA 12.8 + PyTorch 2.11
仍作为独立的 `v0.1.x` 兼容路线维护。PyTorch CUDA 构建、toolkit、FlashInfer/
FlashQLA 构建和启动 profile 必须保持一致，不能混用运行时假设。

**还有哪些硬件风险？**

注意散热、供电稳定性，以及模型和编译缓存所需的 SSD 空间。长 prefill 或反复
CUDA Graph/AOT 编译时降频很容易被误判为软件性能回退。

## 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双 2080 Ti
  模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。本仓库聚焦于补丁后
  的 vLLM runtime。

## 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向 fork，
遵循 Apache-2.0 license，保留上游项目结构，并加入面向双 2080 Ti 的 SM75 runtime
补丁、启动 profile 和验证记录。

当前使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving 框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：attention、sampling
  和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 Gated DeltaNet /
  Qwen hybrid linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  SM70/SM75 适配版本，用于已验证的 Qwen prefill 路线。
- TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM 相关 kernel。

上游更新合入后，仍会在本 fork 的 SM75 范围内重新验证。
