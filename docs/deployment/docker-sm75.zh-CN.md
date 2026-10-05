# SM75 Docker 镜像

语言：[English](docker-sm75.md) | 简体中文

面向 **RTX 2080 Ti（SM75）** 的 vLLM 2080 Ti Definitive Edition 容器镜像，
拉取即可运行。用法与官方 vLLM 镜像一致：入口是 `vllm serve`，模型和参数直接写在
镜像名后面即可。

## 状态

已在**两张 RTX 2080 Ti（2× 22 GiB，tensor parallel 2）**上验证：

- 27B 生产形态模型同时服务文本与多模态请求，CUDA Graph 正常捕获，KV 池
  7.75 GiB，每卡占用约 18.8 GiB，全程 **0 条 NVRM 报错**；冷加载约 13 分钟。
- 小文本模型（`Qwen/Qwen3-0.6B`，FP16）也能正常启动并正确作答。

本硬件上**尚未测量**：吞吐、TTFT/TPOT、长上下文下的 KV 容量、重启后的缓存复用。
容量与吞吐取决于模型、KV 精度、上下文长度与拓扑，来自其他配置的数字不能直接沿用。
NVLink 拓扑以及其他模型、精度、后端同样未验证。

宿主机必须能够分配锁页内存，见[宿主机要求](#宿主机要求)。

CI 在**无 GPU** 的机器上构建镜像（约 **26 分钟**），推送后再按 digest 拉回并测试
这些字节。因此 CI 变绿说明发布产物完整，**不代表 GPU 推理已经验证过**。

## 宿主机要求

只有可用的 NVIDIA 驱动和 GPU 透传还不够：宿主机必须能分配**锁页内存**。否则任何模型
都起不来，表现为

    torch.AcceleratorError: CUDA error: OS call failed or operation not supported on this OS

而内核日志会给出真正的原因：

    NVRM: Failed to create a DMA mapping!
    NVRM: osIovaMap: failed to map allocation (status = 0x59)

这两行来自 `nv_dma_map_alloc` 把内存页交给内核 DMA API 的过程，因此故障在宿主机的
IOMMU/DMA 层——与本镜像、启动参数、模型都无关。官方 vLLM 镜像在这类宿主机上同样
失败。可以在**不经 Docker、不经 vLLM** 的情况下直接验证：

```bash
python3 -c "
import ctypes
cu = ctypes.CDLL('libcuda.so.1')
rc = cu.cuInit(0)
assert rc == 0, ('cuInit failed', rc)
for mb in (64, 128, 256):
    buf = ctypes.c_void_p()
    rc = cu.cuMemAllocHost(ctypes.byref(buf), mb * 1024 * 1024)
    print(mb, 'MB: CUresult', rc, 'OK' if rc == 0 else 'FAIL')
"
```

`CUresult` 为 `0` 表示成功。较大尺寸返回 `304`（`CUDA_ERROR_OPERATING_SYSTEM`）
才是上面说的 DMA 映射失败；返回其他错误码说明原因不同，不要据此去改 IOMMU。若较大
尺寸以 `304` 失败，可在 `/etc/default/grub` 的 `GRUB_CMDLINE_LINUX_DEFAULT` 中加入
`iommu.passthrough=1`，然后 `sudo update-grub && sudo reboot`。重测前先确认参数生效
——日志必须显示 `Passthrough`：

```bash
journalctl -k -b | grep 'Default domain type'
```

注意：`intel_iommu=pt` 在当前内核上**不是合法参数**，传入只会打印 "Unknown option"
并静默失效；真正生效的是 `iommu.passthrough=1`。这是宿主机侧的改动：若宿主机的 IOMMU
翻译域无法映射这类分配，换任何 vLLM 镜像都会同样失败。

## 快速开始

拉取已发布的镜像。`weicj/vLLM-2080Ti-Definitive` 发布到
`ghcr.io/weicj/vllm-2080ti-definitive`；fork 则在各自的 owner 下发布：

```bash
export IMAGE="ghcr.io/weicj/vllm-2080ti-definitive:main"
docker pull "$IMAGE"
```

### Docker Compose（推荐）

```bash
mkdir vllm-sm75 && cd vllm-sm75
curl -fsSL -o compose.yaml https://raw.githubusercontent.com/weicj/vLLM-2080Ti-Definitive/main/docker/docker-compose.sm75.yml
```

该文件也在仓库内：
[`docker/docker-compose.sm75.yml`](../../docker/docker-compose.sm75.yml)。

启动前先配置。`MODELS_DIR` 是宿主机存放模型的目录，`MODEL_NAME` 是它下面的模型
目录名。两者都必填——缺失时 compose 会直接报错，避免你还没选好模型就已经跑起来。

```bash
cat > .env <<'EOF'
MODELS_DIR=/path/to/models
MODEL_NAME=Qwen3-0.6B
EOF
docker compose up -d
```

可选变量：`IMAGE`、`MODEL_ALIAS`、`TP_SIZE`、`MAX_MODEL_LEN`、
`GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS`。其他 vLLM 参数——`--api-key`、
`--kv-cache-dtype`、`--enable-prefix-caching` 等——自己加到 compose 文件的
`command:` 里即可，镜像入口已经是 `vllm serve`。首次启动需要加载权重、并可能触发
内核编译，请留出时间：

```bash
docker compose ps
docker compose logs -f
```

### Docker

```bash
docker run --rm --name vllm-sm75 --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  "$IMAGE" /models/Qwen3-0.6B \
  --dtype half --max-model-len 2048
```

直接传模型和参数即可，**不要再写 `vllm serve`**，入口已包含。

在另一个终端确认服务正常：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

`--ipc=host` 是官方镜像文档的用法，多卡 tensor parallel 也需要它（或足够大的
`--shm-size`）。

**WSL2 宿主机**下 vLLM 默认关闭 pinned memory，启动会以
`RuntimeError: UVA is not available` 中止，需加上
`-e VLLM_WSL2_ENABLE_PIN_MEMORY=1`；原生 Linux 不需要。

compose 文件和 `docker run` 示例都只发布到 `127.0.0.1`。除非自行配置，服务本身没有
认证，因此在向本机之外暴露之前请先加上 `--api-key`。

## 双卡

命令不变，把 `--tensor-parallel-size` 设为 2：

```bash
docker run --rm --name vllm-sm75-2gpu --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  "$IMAGE" /models/Qwen3-0.6B \
  --dtype half --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.9
```

确认两张卡都在工作，并执行与上文相同的 `/health`、`/v1/models` 和生成检查：

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

使用 Compose 时，改为把 `--tensor-parallel-size 2` 加到 compose 文件的 `command:` 里。

先用小模型打通链路，再换成真正需要双卡的模型。请在确切主机上记录：两卡之间是
NVLink 桥还是仅走 PCIe、冷启动耗时、首次 JIT 编译耗时、使用相同缓存卷的第二次
启动耗时、每卡显存占用，以及 `--tensor-parallel-size 2` 下输出是否仍然正确。
双卡服务现已验证通过（见"状态"），但上述数字是逐机器记录的，仍不能说明其他模型、
精度或上下文的容量与吞吐。

## 镜像内容

- **仅 Linux amd64**，CUDA 13.0.3 devel、Ubuntu 24.04、`/opt/venv` 中的
  Python 3.12。这比项目原生主机目标（Ubuntu 26.04+、kernel 7+、GCC 15）更新，
  是一条独立且范围仍然较窄的路线。
- **从本仓库源码构建**的 wheel，而非上游预编译 vLLM wheel，并应用了本 fork 的
  Torch E8M0 补丁。
- FlashQLA 采用固定的
  [`weicj/FlashQLA-SM70-SM75` 源码](https://github.com/weicj/FlashQLA-SM70-SM75/tree/3ab27d77d8ca01d7a4718903b726add1a8886c0e)
  加本仓库 SM75 补丁，位于 `/opt/FlashQLA`，通过 `PYTHONPATH` 及
  `FLASHQLA_ROOT` / `FLASHQLA_DIR` 暴露。这是源码集成，不是另行安装带旧依赖固定
  版本的 FlashQLA distribution。
- 镜像保留 CUDA 工具链、C++ 编译器、Ninja 和头文件，因为 FlashQLA 与 FlashInfer
  会在**首次使用 GPU 时**编译内核。镜像内没有 AOT 预编译，构建阶段也不执行任何
  GPU 代码。
- `/opt/vllm-tools/` 存放发布元数据、`pyproject.toml` 和 Torch E8M0 检查器，便于
  核验。

宿主机仍需要可用的 NVIDIA 驱动和 GPU 透传（NVIDIA Container Toolkit）；容器不能
替代驱动。本路线不支持 ARM/QEMU 或其他 GPU 架构。

再分发镜像时请保留署名与许可证：[上游 vLLM](https://github.com/vllm-project/vllm)
（Apache-2.0）、**vLLM 2080 Ti Definitive Edition**、作者
[`github.com/weicj`](https://github.com/weicj)、
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)、其
[SM70/SM75 适配版本](https://github.com/weicj/FlashQLA-SM70-SM75) 及其他内含第三方
组件。

## 自行构建

通常不需要，CI 会发布镜像。本地构建：

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.sm75 --target vllm-openai \
  --build-arg MAX_JOBS=2 --build-arg NVCC_THREADS=1 \
  --load -t vllm-sm75:local .
export IMAGE=vllm-sm75:local
```

这会从源码编译 CUDA 内核和 Rust 前端，需要较多内存、磁盘和时间。CI 把编译并行度
压得较低并设置约 350 分钟上限，这两点都不构成你本机一定能完成的保证。构建阶段还
需要能对固定的 FlashQLA 仓库执行 `git`。

## 标签与发布

镜像发布到 GHCR，命名空间为小写的仓库名
（`ghcr.io/<owner>/vllm-2080ti-definitive`）：

| 事件 | 检查通过后发布的标签 |
| --- | --- |
| PR，或推送到非默认分支 | 无 —— 仅构建与 smoke |
| 推送到默认分支 | `main` 与 `sha-<commit>`；不移动 `latest` |
| 已发布的 GitHub Release | Release 标签，如 `v0.2.2-post3`；仅非 prerelease 才更新 `latest` |
| 手动触发 | 默认仅构建；要求发布时产生唯一的 `test-…` 标签 |

稳定性以 GitHub 的 prerelease 标记为准，因此 `post3` 不会被通用 semver 规则误判为
prerelease。仅推送 Git tag 不会发布任何镜像。PR 不会获得包写入凭据，也不会被自动
送到个人 GPU runner 上。

GHCR 新包可能默认私有：可能需要先认证，或显式设置包可见性。核验镜像的 OCI
`source` 与 `revision` 标签；可复现部署优先使用 digest 而非标签。

## 验证镜像

以下检查不会启动 CUDA，也不加载原生内核，因此在没有 GPU 的机器上同样可用：

```bash
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py
docker run --rm --entrypoint uv "$IMAGE" pip check --python /opt/venv/bin/python
```

第一条检查 wheel 版本、Torch/CUDA 版本、`import vllm`、Torch E8M0 补丁、FlashQLA
源码与补丁、预期路径以及 JIT 工具；第二条检查依赖集合是否自洽。

这里不能直接使用 `vllm serve --help`：没有 GPU 时它会在检测设备阶段失败。下面的
独立 CLI 检查会在隔离进程中运行真实安装的 console script，仅补上缺失的设备元数据，
验证的是 CLI 导入与参数解析，**不是 GPU 启动**：

```bash
docker image inspect --format '{{json .Config.Entrypoint}}' "$IMAGE"
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py --cli-help
```

预期入口为 `["vllm","serve"]`。在真实 GPU 主机上，还可以用
`docker run --rm --gpus all "$IMAGE" --help` 检查未经改动的入口。

要执行真实内核，需显式传入 `--gpu`。GPU 缺失或不是 SM75 时它会直接报错而不会跳过，
且只测试 device 0：

```bash
docker run --rm --gpus all --ipc=host \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  --entrypoint python "$IMAGE" /opt/vllm-tools/smoke_docker_sm75.py --gpu
```

它检查 vLLM 核心 SiLU 操作、FlashInfer CUDA-core decode 与 FP32 参考结果，以及
FlashQLA JIT 的 `gdn_forward` / `gdn_forward_varlen`（含原生 FP16 Q/K/V 与 FP32
的对照、packed 与独立序列的对照）。使用的是极小张量，不是 benchmark、模型下载、
后端资格验证或 tensor parallel 测试。

首次 JIT 成功后，用相同缓存卷在新容器中再执行一次以检查缓存复用，并**分别记录两次
运行**，因为仅再次运行成功不能说明没有重新编译。
`TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` 是 FlashQLA 的持久化缓存。
不兼容的镜像或工具链更新应使用不同缓存，也不要把宿主机虚拟环境挂载覆盖 `/opt/venv`。

## 运行真实模型

服务自己的权重是常规做法：只读挂载目录，然后传容器内的路径，见"快速开始"。
`--served-model-name` 用于设置客户端看到的模型 ID。

如果改为让 vLLM 从 HuggingFace Hub 下载权重，请挂载缓存目录，避免每次重建都重新
下载；仓库需要授权时传 `HF_TOKEN`——不要把它烘焙进镜像：

```bash
docker run --rm --name vllm-sm75-hub --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN "$IMAGE" Qwen/Qwen3-0.6B \
  --dtype half --max-model-len 2048 \
  --host 0.0.0.0 --port 8000
```

启动后检查 health、实际 served model ID 以及非空的真实生成结果——不能只看 HTTP
状态码：

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
curl --fail --silent --show-error http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

示例中的 `Qwen/Qwen3-0.6B` 只是为快速加载而选的小尺寸模型，不是推荐部署 Profile。

Qwen3.5 系列使用本 fork FlashQLA 后端的 GDN 路线，需要显式传入
`--gdn-prefill-backend flashqla_legacy`（不是 `--mamba-backend flashqla_legacy`），
并与 `--dtype half` 及其余路线设置一起使用。Qwen3-0.6B smoke 不会执行 GDN。

镜像不运行 `launcher.sh`，也不应用 Profile：GPU 编号、tensor parallel 数、上下文、
MTP、reasoning 及 chat-template 都由你传入。要复现原生部署，请用
`launcher.sh --print-config` 读出目标 launcher 的有效配置，再转换为参数与环境变量；
各路线的说明见[硬件 Profile 指南](../../profiles/README.zh-CN.md)。复用任何原生
主机性能数字前，先确认其对应的模型、精度、MTP 设置、上下文、拓扑与 benchmark 口径。