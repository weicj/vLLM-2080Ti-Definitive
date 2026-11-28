<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition 题图](docs/assets/vllm-2080ti-definitive-title.jpeg)

语言：[English](README.md) | 简体中文

面向双 RTX 2080 Ti 及其他 SM75 GPU 的 vLLM 专用推理运行时，包括 Tesla
T10/T40/T4、TITAN RTX 和 Quadro RTX 6000/8000。

这个硬件定向 fork 维护复现上述 Turing 推理栈所需的 SM75 补丁、launcher
profile 和验证资料。它基于上游 vLLM。再发布派生版本时必须保留上游许可证、上游署名以及
`github.com/weicj` 的项目署名。

欢迎在 [Discord 交流群](https://discord.gg/VFqVVySdMS) 分享使用体验、提出功能请求，
以及交流部署和运行问题。

![0.2.x 单请求推理速度演示](docs/assets/vllmspeed_dflash.gif)

当前版本为 `v0.3.0`，分支为 `0.3.x`。

仓库维护 SM75 补丁、FlashQLA 扩展、启动脚本和硬件 profile。

0.3.x 分支起，使用 Docker 部署。需要宿主机原生构建的用户可使用 [0.2.x 发行版](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.2-post3)。

版本记录：[CHANGELOG.md](CHANGELOG.md)

## 💡 主要支持架构

本项目的运行时覆盖 NVIDIA Turing（SM75，CUDA 计算能力 7.5）全系列显卡。
可按单卡显存和卡数选用已有实测的 Profile 作为起点。

- **单卡 16 GiB** 包括 Tesla T10、Tesla T4 和 Quadro RTX 5000。按卡数参考 T10 的配置指南。[2 x T10](profiles/2xT10/README.zh-CN.md) · [4 x T10](profiles/4xT10/README.zh-CN.md)
- **单卡 24 GiB** 包括 TITAN RTX、Quadro RTX 6000 和 24 GiB 版 Tesla T40。双卡可参考 [2 x 2080 Ti Profile 指南](profiles/2x2080Ti/README.zh-CN.md)。

| 指标 | 2 x 2080Ti 22G | 2 x T10 16G | 4 x T10 16G |
| --- | ---: | ---: | ---: |
| 张量并行 | TP2 | TP2 | TP4 |
| CUDA 核心总数 | 8,704 | 7,168 | 14,336 |
| Tensor Core 总数 | 1,088 | 896 | 1,792 |
| 理论稠密 FP16 Tensor 算力 | 227.7 TFLOPS | 182.4 TFLOPS | 364.7 TFLOPS |
| 总显存 | 44 GiB | 32 GiB | 64 GiB |
| 总显存带宽（理论） | 1,232 GB/s | 806.5 GB/s | 1,613.1 GB/s |
| Ring All-Reduce 理论累计带宽 | 100 GB/s（NVLink NV2） | 31.5 GB/s（PCIe 3.0 x16 P2P） | 63.0 GB/s（PCIe 3.0 x16 P2P） |
| Qwen3.8-27B 每解码 token 的总归约流量 | 2.50 MiB | 2.50 MiB | 7.50 MiB |
| 每 token 的链路传输时间下限 | 26.2 µs | 83.2 µs | 124.8 µs |
| 每 token 的 FP16 归约加法时间下限 | 23 ns | 29 ns | 43 ns |
| 推荐配置 | [Profile 指南](profiles/2x2080Ti/README.zh-CN.md) | [Profile 指南](profiles/2xT10/README.zh-CN.md) | [Profile 指南](profiles/4xT10/README.zh-CN.md) |

## ⚡ 性能亮点

### Qwen3.8-27B-NVFP4

| 硬件 | 上下文 | 测速并发 | 4K/128 吞吐 | 32K/512 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 2 x 2080Ti 22G | **1 x 262K** | 1 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| 4 x T10 16G | **4 x 220K** | 1 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| 4 x T10 16G | **4 x 220K** | 4 | 1585.96 / 797.18 | 1540.44 / 711.94 |

### Inkling 2 Mini

| 硬件 | 上下文 | 测速并发 | 4K/128 吞吐 | 32K/512 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 2 x 2080Ti 22G | **1 x 870K** | 1 | 2793.98 / 275.03 | 2580.39 / 207.40 |
| 2 x 2080Ti 22G | **4 x 140K** | 4 | 2736.95 / 594.34 | 2427.26 / 369.16 |
| 2 x T10 16G | **1 x 262K** | 1 | 2465.67 / 237.90 | 2418.85 / 174.58 |
| 2 x T10 16G | **2 x 176K** | 2 | 2370.57 / 384.04 | 2108.90 / 238.03 |
| 4 x T10 16G | **1 x 1M** | 1 | 3729.50 / 290.63 | 3480.09 / 232.84 |
| 4 x T10 16G | **2 x 512K** | 2 | 3398.49 / 444.66 | 3353.87 / 338.78 |
| 4 x T10 16G | **4 x 250K** | 4 | 2966.63 / 653.72 | 2992.04 / 427.90 |

吞吐按 prefill / decode 显示，单位为 tok/s。并发解码统计共同解码窗口内的总吞吐。
以上数据均使用 DFlash2（`n=7`），在草稿持续高接受率的合成输入下测得。真实任务中
草稿接受率下降时，解码吞吐可能低于表中结果。

## 🚀 构建与启动

1. 克隆 0.3.x 分支并构建 Docker 镜像。

   ```bash
   git clone --branch 0.3.x https://github.com/weicj/vLLM-2080Ti-Definitive.git
   cd vLLM-2080Ti-Definitive
   ./build.sh
   ```

   Docker 构建使用官方镜像提供的 Torch 和 CUDA，
   [镜像配置](PROJECT_RELEASE.env) 用 digest 锁定官方 linux/amd64 镜像。
   构建时应用 SM75 补丁、编译 FlashQLA，并在同一镜像中加入 Inkling 2 Mini 支持。

   构建镜像需要启用 BuildKit 的 Docker 和 Buildx 插件。运行镜像需要
   NVIDIA Container Toolkit 和兼容 CUDA 13.0 的 NVIDIA 驱动。
   Python、PyTorch 和 CUDA 用户态依赖由镜像提供。

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

![0.2.x launcher.sh 交互式主菜单](docs/assets/launcher-main-menu.png)

自动化部署可使用非交互参数：

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-INT4 \
PROFILE=2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=0,1 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

使用 `./launcher.sh --print-config` 预览路线。自动化部署见
[非交互启动说明](docs/non-interactive-launch.zh-CN.md)。

## 🧭 Profile 与推荐路线

目录结构和路线字段见 [Profile 指南](profiles/README.zh-CN.md)。

有多个权重版本的模型使用 `profiles/<硬件>/<模型>/<权重>/`。
启动模式默认为 `MODE=fast`，可以由 launcher 或 profile 显式设置。

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

0.3.x 使用官方镜像内的 CUDA 13.0 和 PyTorch。宿主机需要兼容的 NVIDIA 驱动、Docker、NVIDIA Container Toolkit 和 Python 3。性能表记录了具体的测试版本。

**还有哪些硬件风险？**

注意散热、供电稳定性，以及模型和编译缓存所需的 SSD 空间。长 prefill 或反复
CUDA Graph/AOT 编译时降频很容易被误判为软件性能回退。

## 🔗 相关项目

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox)：双 2080 Ti
  模型路线、benchmark 汇总、模型记录和运行建议的配套工具箱。本仓库聚焦于补丁后
  的 vLLM runtime。

## 🙏 致谢 / 上游项目

本仓库是基于上游 vLLM 的硬件定向 fork，遵循 Apache-2.0 许可证，提供 SM75 运行时
补丁、启动 profile 和验证资料。[上游 vLLM](https://github.com/vllm-project/vllm)

当前使用或集成的加速组件包括：

- [vLLM](https://github.com/vllm-project/vllm)：基础推理引擎和 serving 框架。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：attention、sampling
  和量化 kernel 路线。
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)：上游 Gated DeltaNet /
  Qwen hybrid linear-attention 实现。
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)：
  SM70/SM75 适配版本，用于已验证的 Qwen prefill 路线。
- TurboQuant、Marlin、CUTLASS、Triton 以及 vLLM 相关 kernel。

本项目将 Inkling 2 Mini 的上游实现移植到 vLLM v0.34.0，并采用 lvb-s 的 SM75 适配。
[上游实现](https://github.com/vllm-project/vllm/pull/69028) ·
[社区 SM75 移植](https://github.com/lvb-s/vllm-inkling-sm75)

上游更新合入后，仍会在本 fork 的 SM75 范围内重新验证。
