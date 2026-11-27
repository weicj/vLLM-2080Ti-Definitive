<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-definitive-title.jpeg)

语言：[English](README.md) | 简体中文

面向双 RTX 2080 Ti 及其他 SM75 GPU 的 vLLM 专用推理运行时，包括 Tesla
T10/T40/T4、TITAN RTX 和 Quadro RTX 6000/8000。

这个硬件定向 fork 保留了复现上述 Turing 推理栈所需的 SM75 专用源码修改、launcher
profile 和验证资料。它基于上游 vLLM；再发布派生版本时必须保留上游许可证、上游署名以及
`github.com/weicj` 的项目署名。

欢迎在 [Discord 交流群](https://discord.gg/VFqVVySdMS) 分享使用体验、提出功能请求，
以及交流部署和运行问题。

当前开发版本为 `v0.3.0.dev0`，分支为 `vllm-2080ti-definitive-0.3.x`。
Docker 运行时使用官方 [vLLM v0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)。
本轮的实测版本、测试环境和性能数据记录在各硬件的 profile 说明中。

版本记录：[CHANGELOG.md](CHANGELOG.md)

## 💡 为什么用 RTX 2080 Ti 做 LLM 推理？

这个项目的判断很实际：两张通过 NVLink 连接的 22 GB RTX 2080 Ti 提供 44 GB
显存、较高的显存带宽和 136 个 Turing SM。经过针对 SM75 的 vLLM 适配后，这套
硬件不只是运行小模型，也足以承载严肃的本地 27B 与 35B 级模型服务。

| 指标 | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB 基线 | 倍率 |
| --- | ---: | ---: | ---: |
| 专用 FP32 datapath | 8,704 | 5,376 | 1.62x |
| SM 数量 | 136 | 84 | 1.62x |
| Tensor Core | 1,088 | 336 | 3.24x |
| Dense FP16 矩阵吞吐 | 228 TFLOPS | 160 TFLOPS | 1.43x |
| 总显存带宽 | 1,232 GB/s | 1,008 GB/s | 1.22x |
| 总显存 | 44 GB | 24 GB | 1.83x |

本 fork 通过 Marlin、FlashInfer/FlashQLA、TurboQuant/INT8 KV、MTP/DFlash2 和 CUDA
Graph，把这些硬件资源转成可用的 serving 栈。

## 🧩 支持状态

原生构建的目标环境是 Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15、CUDA 13.0 与 PyTorch 2.13。如果使用 CUDA 12.8、PyTorch 2.11、较早 kernel 或 GCC 12/13/14，可参考已不再持续维护的 `0.1.x` 版本线。

支持的模型路线和实测数据以对应硬件组的 profile 说明为准。

Launcher 支持 TP、PP 和 TP/PP 混合推理。0.3.x Docker 已完成双 T10（TP=2）和四 T10（TP=4）的 Qwen 与 Inkling 验证。

## 🧪 已验证模型与 Profile

| 硬件 | 模型 | 权重 | Profile 说明 |
| --- | --- | --- | --- |
| 2×T10 | Qwen3.8-27B | INT4（W4A16） | [Qwen3.8-27B-INT4](profiles/2xT10/README.zh-CN.md#qwen38-27b-int4) |
| 4×T10 | Qwen3.8-27B | FP8（W8A16）、NVFP4（W4A16） | [FP8](profiles/4xT10/README.zh-CN.md#qwen38-27b-fp8) · [NVFP4](profiles/4xT10/README.zh-CN.md#qwen38-27b-nvfp4) |
| 2×T10、4×T10 | Inkling 2 Mini | 官方权重 | [2×T10 纯 AR 与 DSpark/7](profiles/2xT10/README.zh-CN.md#inkling-2-mini) · [4×T10 纯 AR 与 DSpark/7](profiles/4xT10/README.zh-CN.md#inkling-2-mini) |

## 🚀 构建与启动

1. 在 0.3.x 工作树中运行 `./build.sh`。默认构建使用官方镜像提供的 Torch 和 CUDA，
   [镜像配置](PROJECT_RELEASE.env) 用 digest 锁定官方 linux/amd64 镜像。
   构建时应用 SM75 GDN prefill 补丁，并将编译好的 FlashQLA 扩展装入镜像。

   宿主机需要 Docker、NVIDIA Container Toolkit 和兼容 CUDA 13.0 的 NVIDIA 驱动。
   原生构建和启动仍可分别通过 `VLLM_RUNTIME_BACKEND=native ./build.sh` 和
   `VLLM_RUNTIME_BACKEND=native ./launcher.sh` 使用。

2. 运行 `./update.sh` 更新。Git 工作树对跟踪分支执行快进更新，归档安装查找稳定的
   0.3.x Release。脚本保留本地缓存、虚拟环境、日志和 `profiles/local`，
   更新后询问是否运行 `build.sh`。

3. 启动和管理服务：

```bash
./launcher.sh
```

交互式 launcher 可选择 target 与 DFlash draft 权重、应用 Profile、设置 GPU 和
TP/PP 拓扑、选择启动模式与网络配置，并在启动时自动完成健康检查和 smoke 测试；
也可以在菜单中停止已启动的服务。

自动化部署可使用非交互参数：

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-INT4 \
PROFILE=2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=0,1 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

使用 `./launcher.sh --print-config` 预览路线。自动化部署见
[非交互启动说明](docs/non-interactive-launch.zh-CN.md)。

Docker launcher 继续使用本地权重路径、profile、GPU 顺序和服务配置。
`fast` 模式采用上游调度与 CUDA Graph 默认配置。`normal` 模式的投机解码和双卡
T10 的 Qwen TurboQuant MTP 图文路线使用 PIECEWISE。服务级思考预算和
prefill barrier 可通过原生运行时使用。

### 在 SM75 上运行 Inkling 2 Mini

Inkling 镜像把 v0.34.0 发布后合入主线的模型支持移植到固定底座。先构建镜像，
再通过 `PROFILE` 选择路线。下面的示例使用支持 PCIe P2P 的两张 T10。

```bash
VLLM_DOCKER_VARIANT=inkling ./build.sh

VLLM_DOCKER_VARIANT=inkling MODEL_DIR=/path/to/Inkling-2-Mini \
PROFILE=2xT10/inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env \
GPU_DEVICES=0,1 TP_SIZE=2 NCCL_P2P_DISABLE=0 DISABLE_PREFIX_CACHING=1 \
NON_INTERACTIVE=1 ./launcher.sh
```

纯 AR 请选择 `nomtp-fp8kv-1x8K-text-only.env`。四卡使用 `4xT10` 下的同名
profile，将 `TP_SIZE` 设为 `4`，并选择四张支持 PCIe P2P 的 GPU。

该实验镜像使用 vLLM 上游模型实现和 lvb-s 的 SM75 移植。
[官方权重](https://huggingface.co/thinkingmachines/Inkling-2-Mini) ·
[上游实现](https://github.com/vllm-project/vllm/pull/69028) ·
[社区 SM75 移植](https://github.com/lvb-s/vllm-inkling-sm75)

## 🧭 Profile 与推荐路线

目录结构和路线字段见 [Profile 指南](profiles/README.zh-CN.md)。
双 RTX 2080 Ti 的原生路线见 [2x2080Ti Profile](profiles/2x2080Ti/README.zh-CN.md)。

Profile 按扁平路径 `profiles/<硬件>/<模型>/<权重>/<路线>.env` 组织。启动模式由
launcher 选择，默认 `MODE=fast`；也可以由 launcher 或 profile 显式设置。

可用模式：

- `normal`：稳定的日常部署模式。
- `fast`：用于已验证路线的更高性能模式，也是默认模式。
- `aggressive`：性能最高，但质量风险也最高。
- `safe`：用于排障和兼容性的保守回退模式。

Profile 只选择路线参数。GPU、端口、target 与 draft 模型路径、chat template 和
reasoning 默认值由 launcher 统一管理。

## 🛠️ 目标硬件

- 两张经 NVLink 连接的 RTX 2080 Ti 22 GB
- 两张通过 PCIe 连接的 16 GiB Tesla T10（TP=2 Profile）
- 四张通过 PCIe 连接的 16 GiB Tesla T10（TP=4 Profile）
- NVIDIA Turing / SM75，已验证的 tensor parallel size 为 2 和 4
- 原生运行时目标：CUDA 13.0、PyTorch 2.13、Python 3.12
- 原生构建主机：Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15

其它 Turing 显卡仍需针对显存容量、PCIe/NVLink 拓扑、模型 head dimension、
KV cache dtype 和 CUDA Graph 行为独立验证。

## ❓ 硬件 Q&A

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

原生运行时使用 CUDA 13.0 + PyTorch 2.13。使用旧的 CUDA 12.8 + PyTorch 2.11
时，可参考已停止维护的 `v0.1.x` 兼容路线。PyTorch CUDA 构建、toolkit、FlashInfer/
FlashQLA 构建和启动 profile 必须保持一致，不能混用运行时假设。

**还有哪些硬件风险？**

注意散热、供电稳定性，以及模型和编译缓存所需的 SSD 空间。长 prefill 或反复
CUDA Graph/AOT 编译时降频很容易被误判为软件性能回退。

## 🔗 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双 2080 Ti
  模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。本仓库聚焦于补丁后
  的 vLLM runtime。

## 🙏 致谢 / 上游项目

本仓库是基于上游 [vLLM](https://github.com/vllm-project/vllm) 的硬件定向 fork，
遵循 Apache-2.0 license，保留上游项目结构，并加入面向双 2080 Ti 的 SM75 runtime
补丁和启动 profile。

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
