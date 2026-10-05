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

![单请求实时测速演示](docs/assets/vllmspeed_dflash.gif)

当前 0.2.x 基线：`v0.2.2-post3`
上游基线：`b23433088b`（`v0.29.1rc0-33`）

分支：[`main`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/main)
版本参考：[v0.2.2-post3](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.2-post3)
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

支持的硬件还包括两张通过 PCIe 连接的 16 GiB Tesla T10，提供已验证的 TP=2
Qwen 27B 路线。四卡 Tesla T10 Profile 面向 TP=4 服务；存在图像语义失败的路线
会在硬件 README 中明确标记为 candidate。

## 🧩 支持状态

当前目标环境是 Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15、CUDA 13.0 与 PyTorch 2.13。如果使用 CUDA 12.8、PyTorch 2.11、较早 kernel 或 GCC 12/13/14，可参考已不再持续维护的 `0.1.x` 版本线。

支持的模型路线和实测数据以对应硬件组的 profile 说明为准。

Launcher 支持 TP、PP 及 TP/PP 混合推理；已验证布局包括双 RTX 2080 Ti（TP=2）、双 Tesla T10（TP=2）和四张 Tesla T10（TP=4）。

## 🧪 已验证模型路线

当前实测模型和权重路线：

| 模型路线 | 权重路线 | 模型卡 | 推荐场景 | Profile 路径 |
| --- | --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 高精度单并发 | `qwen27b/w8a16` |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 长上下文多并发 | `qwen27b/w4a16` |
| Qwen3.8 27B | INT4 (W4A16) | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | 2xT10 MTP3 推理 | `qwen27b/w4a16` |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | 个人快速推理 | `qwen35b/w8a16` |

## ⚡ 性能亮点

| 硬件 | 权重 | 上下文 / KV | 4K 输入解码 | 32K 输入解码 |
| --- | --- | --- | ---: | ---: |
| 2x RTX 2080 Ti | Qwen3.8 27B NVFP4 | 256K / FP8 KV | **220.84 tok/s** | **209.35 tok/s** |
| 4x Tesla T10 | Qwen3.8 27B FP8 | 256K / FP16 KV | **191.89 tok/s** | **189.38 tok/s** |

两组均为单请求测试，使用 DFlash2（默认 K=7）和高投机命中率的纯文本合成输入；真实任务吞吐会受到 draft 接受率影响，可能无法达到以上数据。

## 🚀 构建与启动

1. 首次构建：

```bash
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd vLLM-2080Ti-Definitive
./build.sh
```

2. 将已有仓库更新到最新 GitHub Release：

```bash
./update.sh
```

更新脚本会保留本地虚拟环境、依赖缓存、日志、结果以及 `profiles/local`，比较本地
`VERSION` 与最新 Release，下载对应源码归档，并在更新完成后询问是否立即运行
`build.sh`。

3. 启动和管理服务：

```bash
./launcher.sh
```

交互式 launcher 可选择 target 与 DFlash draft 权重、应用 Profile、设置 GPU 和
TP/PP 拓扑、选择启动模式与网络配置，并在启动时自动完成健康检查和 smoke 测试；
也可以在菜单中停止已启动的服务。

![launcher.sh 交互式主菜单](docs/assets/launcher-main-menu.png)

自动化部署可使用非交互参数：

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=1,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

使用 `./launcher.sh --print-config` 预览路线。自动化部署见
[非交互启动说明](docs/non-interactive-launch.zh-CN.md)。

### Docker

已发布的镜像与官方 vLLM 镜像行为一致：入口为 `vllm serve`，模型和参数直接写在
镜像名后面即可。`weicj/vLLM-2080Ti-Definitive` 发布到
`ghcr.io/weicj/vllm-2080ti-definitive`。

#### Docker Compose（推荐）

```bash
mkdir vllm-sm75 && cd vllm-sm75
curl -fsSL -o compose.yaml https://raw.githubusercontent.com/weicj/vLLM-2080Ti-Definitive/main/docker/docker-compose.sm75.yml
```

然后先配置。`MODELS_DIR` 是宿主机存放模型的目录，`MODEL_NAME` 是它下面的模型目录
名。两者都必填：缺了会直接报错退出，而不是静默用你没选过的配置跑起来。

```bash
cat > .env <<'EOF'
MODELS_DIR=/path/to/models
MODEL_NAME=Qwen3-0.6B
EOF
docker compose up -d
```

可选变量：`IMAGE`、`MODEL_ALIAS`、`TP_SIZE`、`MAX_MODEL_LEN`、
`GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS`。需要其他 vLLM 参数时，自己加到 compose
文件的 `command:` 里即可——镜像入口已经是 `vllm serve`。

查看日志并确认服务正常（想持续跟踪日志就给 logs 加上 `-f`）：

```bash
docker compose logs --tail=100
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

更新到最新镜像：

```bash
docker compose pull && docker compose up -d
```

#### Docker

```bash
docker run -d --name vllm-sm75 --restart unless-stopped --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/weicj/vllm-2080ti-definitive:main \
  /models/Qwen3-0.6B --dtype half --tensor-parallel-size 2
```

按上面的方式同样确认。除非自行配置，服务本身没有认证，因此 compose 文件里也注明
了：对外暴露前先加 `--api-key`。

两种方式都只运行已发布镜像，都不会构建镜像。双卡 tensor parallel、缓存挂载、
镜像验证以及 WSL2 说明见双语
[SM75 Docker 指南](docs/deployment/docker-sm75.zh-CN.md)。
`docker/Dockerfile.sm75` 从本仓库源码构建镜像（仅 SM75、Linux amd64、CUDA
13.0.3、Ubuntu 24.04、Python 3.12），并应用 Torch 与 FlashQLA 补丁；它与沿用上游的
`docker/Dockerfile` 相互独立，原生构建、launcher 和 Profile 均未改动。

**已在两张 RTX 2080 Ti 上验证**（27B 生产形态模型、文本与多模态、tensor parallel 2）；
该硬件上的吞吐与容量尚未测量。

## 🧭 Profile 与推荐路线

目录结构和路线字段请参阅 [Profile 指南](profiles/README.zh-CN.md)；详细 Profile
说明与参考性能见 [2x2080Ti](profiles/2x2080Ti/README.zh-CN.md)、
[2xT10](profiles/2xT10/README.zh-CN.md) 和
[4xT10](profiles/4xT10/README.zh-CN.md)。

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
- `0.2.x` 目标：CUDA 13.0、PyTorch 2.13、Python 3.12
- 目标主机：Ubuntu 26.04 及以上、Linux kernel 7 及以上、GCC/G++ 15

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

`0.2.x` 目标是 CUDA 13.0 + PyTorch 2.13。使用旧的 CUDA 12.8 + PyTorch 2.11
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
