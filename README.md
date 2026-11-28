<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-definitive-title.jpeg)

Language: English | [简体中文](README.zh-CN.md)

The definitive vLLM inference runtime for dual RTX 2080 Ti and other SM75 GPUs,
including Tesla T10/T40/T4, TITAN RTX, and Quadro RTX 6000/8000.

This hardware-focused fork maintains the SM75 patches, launcher profiles, and
validation evidence needed to reproduce these Turing inference stacks. It is
based on upstream vLLM. Retain both the upstream license and attribution to
`github.com/weicj` when redistributing a derivative.

For usage feedback, feature requests, and community discussion, join the
[Discord community](https://discord.gg/VFqVVySdMS).

![Single-request throughput demo from 0.2.x](docs/assets/vllmspeed_dflash.gif)

Current version is `v0.3.0` on `0.3.x`.

This repository maintains the SM75 patches, FlashQLA extension, launcher,
and hardware profiles.

Starting with 0.3.x, deployments use Docker. Native builds are available in the [0.2.x release](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.2-post3).

Release history: [CHANGELOG.md](CHANGELOG.md)

## 💡 Primary GPU Architecture

The runtime targets the full NVIDIA Turing GPU family (SM75, CUDA compute
capability 7.5). Choose a tested Profile as a starting point based on VRAM
per GPU and the number of cards.

- **16 GiB per GPU** includes Tesla T10, Tesla T4, and Quadro RTX 5000. Use the
  T10 Profile guide for your card count. [2 x T10](profiles/2xT10/README.md) · [4 x T10](profiles/4xT10/README.md)
- **24 GiB per GPU** includes TITAN RTX, Quadro RTX 6000, and the 24 GiB Tesla T40.
  For a dual-GPU setup, start with the [2 x 2080 Ti Profile guide](profiles/2x2080Ti/README.md).

| Metric | 2 x 2080Ti 22G | 2 x T10 16G | 4 x T10 16G |
| --- | ---: | ---: | ---: |
| Tensor parallelism | TP2 | TP2 | TP4 |
| Total CUDA cores | 8,704 | 7,168 | 14,336 |
| Total Tensor Cores | 1,088 | 896 | 1,792 |
| Theoretical dense FP16 Tensor throughput | 227.7 TFLOPS | 182.4 TFLOPS | 364.7 TFLOPS |
| Total VRAM | 44 GiB | 32 GiB | 64 GiB |
| Theoretical total VRAM bandwidth | 1,232 GB/s | 806.5 GB/s | 1,613.1 GB/s |
| Theoretical aggregate Ring All-Reduce bandwidth | 100 GB/s (NVLink NV2) | 31.5 GB/s (PCIe 3.0 x16 P2P) | 63.0 GB/s (PCIe 3.0 x16 P2P) |
| Qwen3.8-27B total All-Reduce traffic per decoded token | 2.50 MiB | 2.50 MiB | 7.50 MiB |
| Link transfer time lower bound per token | 26.2 µs | 83.2 µs | 124.8 µs |
| FP16 reduction arithmetic lower bound per token | 23 ns | 29 ns | 43 ns |
| Recommended configurations | [Profile guide](profiles/2x2080Ti/README.md) | [Profile guide](profiles/2xT10/README.md) | [Profile guide](profiles/4xT10/README.md) |

## ⚡ Highlights

### Qwen3.8-27B-NVFP4

| Hardware | Context | Test concurrency | 4K/128 Throughput | 32K/512 Throughput |
| --- | ---: | ---: | ---: | ---: |
| 2 x 2080Ti 22G | **1 x 262K** | 1 | 1370.18 / 220.84 | 1264.66 / 209.35 |
| 4 x T10 16G | **4 x 220K** | 1 | 1517.36 / 304.25 | 1424.45 / 275.50 |
| 4 x T10 16G | **4 x 220K** | 4 | 1585.96 / 797.18 | 1540.44 / 711.94 |

### Inkling 2 Mini

| Hardware | Context | Test concurrency | 4K/128 Throughput | 32K/512 Throughput |
| --- | ---: | ---: | ---: | ---: |
| 2 x 2080Ti 22G | **1 x 870K** | 1 | 2793.98 / 275.03 | 2580.39 / 207.40 |
| 2 x 2080Ti 22G | **4 x 140K** | 4 | 2736.95 / 594.34 | 2427.26 / 369.16 |
| 2 x T10 16G | **1 x 262K** | 1 | 2465.67 / 237.90 | 2418.85 / 174.58 |
| 2 x T10 16G | **2 x 176K** | 2 | 2370.57 / 384.04 | 2108.90 / 238.03 |
| 4 x T10 16G | **1 x 1M** | 1 | 3729.50 / 290.63 | 3480.09 / 232.84 |
| 4 x T10 16G | **2 x 512K** | 2 | 3398.49 / 444.66 | 3353.87 / 338.78 |
| 4 x T10 16G | **4 x 250K** | 4 | 2966.63 / 653.72 | 2992.04 / 427.90 |

Throughput is reported as prefill / decode in tok/s. Concurrent decode sums
tokens generated during the shared decode window. All results use DFlash2
(`n=7`) with synthetic inputs that sustain high draft acceptance. Decode throughput
can be lower on real tasks with lower draft acceptance.

## 🚀 Build And Launch

1. Clone the 0.3.x branch and build the Docker image.

   ```bash
   git clone --branch 0.3.x https://github.com/weicj/vLLM-2080Ti-Definitive.git
   cd vLLM-2080Ti-Definitive
   ./build.sh
   ```

   The Docker build takes the Torch and CUDA stack from the upstream release
   image. The
   [image lock](PROJECT_RELEASE.env) pins the official linux/amd64 image by
   digest. The build applies the SM75 patches, compiles FlashQLA, and includes
   Inkling 2 Mini support in the same image used for Qwen.

   Building the image requires Docker with BuildKit and the Buildx plugin.
   Running it requires NVIDIA Container Toolkit and a host driver compatible
   with CUDA 13.0. The image provides the Python,
   PyTorch, and CUDA user-space dependencies.

2. Run `./update.sh` to update. Git checkouts fast-forward their tracking
   branch. Archive installs look for a stable 0.3.x release. The updater
   preserves local caches, virtual environments, logs, and `profiles/local`,
   and offers to run `build.sh` after updating.

3. Start and manage the service:

```bash
./launcher.sh
```

The interactive launcher selects target and DFlash draft checkpoints, applies a
profile, configures GPU and TP/PP topology, chooses the launch mode and network
settings, starts the service with health and smoke checks, and stops a running
service.

![launcher.sh interactive main menu from 0.2.x](docs/assets/launcher-main-menu.png)

For automated deployment, pass the configuration non-interactively:

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-INT4 \
PROFILE=2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=0,1 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

Use `./launcher.sh --print-config` to preview a route. See the
[non-interactive launch guide](docs/non-interactive-launch.md) for automation.

## 🧭 Profiles

The [Profile Guide](profiles/README.md) explains the layout and route fields.

Models with multiple weight variants use `profiles/<hardware>/<model>/<weight>/`.
The launcher selects the startup mode and defaults to `MODE=fast`. The launcher or profile can also set the mode explicitly.

Available modes:

- `normal`: stable daily deployment mode.
- `fast`: higher-performance mode for validated routes; this is the default mode.
- `aggressive`: highest-performance mode with increased quality risk.
- `safe`: conservative fallback for troubleshooting and compatibility.

The profile selects only route parameters. The launcher owns GPU selection,
port, target and draft model paths, chat template, and reasoning defaults.

## 🛠️ Hardware Target

- Two RTX 2080 Ti 22 GB GPUs connected by NVLink
- Two 16 GiB Tesla T10 GPUs over PCIe (TP=2 profiles)
- Four 16 GiB Tesla T10 GPUs over PCIe (TP=4 profiles)
- NVIDIA Turing / SM75, with validated tensor parallel sizes 2 and 4

Other Turing cards need independent validation for VRAM capacity, PCIe/NVLink
topology, model head dimensions, KV-cache dtype, and CUDA Graph behavior.

## ❓ Hardware Q&A

**What GPU interconnect is required?**

NVLink is recommended. PCIe P2P is the baseline requirement, but narrow PCIe
links without NVLink are not a proven substitute for the validated topology.
Confirm P2P and benchmark the actual host topology before treating it as a
deployment route.

**Does the host need a strong CPU or a lot of RAM?**

A high-end CPU is not required, but modern single-core performance and low
platform latency matter. More RAM mainly helps builds, downloads, and compile
cache. Very old CPU platforms can lower decode throughput even when the GPUs
are unchanged.

**Can 11 GB and 22 GB Turing cards be mixed?**

Not for the documented 27B/35B TP=2 routes. Tensor parallelism is effectively
limited by the smaller rank. Better alternatives are paired high-VRAM TU102
cards, such as TITAN RTX, Quadro RTX 6000, or Quadro RTX 8000, with NVLink or
confirmed PCIe P2P and a separate profile validation.

**Which CUDA and PyTorch versions apply?**

The 0.3.x image supplies CUDA 13.0 and PyTorch. The host needs a compatible
NVIDIA driver, Docker, the NVIDIA Container Toolkit, and Python 3. The
profile guides record the exact versions used for their benchmarks.

**What other hardware risks matter?**

Cooling, stable power delivery, and enough SSD capacity for weights and compile
caches. Thermal throttling can look like a software performance regression,
particularly during long prefill and repeated CUDA Graph/AOT compilation.

## 🔗 Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual-2080-Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime.

## 🙏 Credits And Upstream Projects

This repository is a hardware-focused fork of
vLLM, licensed under Apache-2.0. It provides SM75 runtime patches, launch
profiles, and validation notes. [Upstream vLLM](https://github.com/vllm-project/vllm)

Acceleration components used or integrated by this runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream Gated
  DeltaNet / Qwen hybrid linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the validated Qwen prefill route.
- TurboQuant, Marlin, CUTLASS, Triton, and related vLLM kernels.

Inkling 2 Mini support backports the upstream model implementation to vLLM v0.34.0
and uses the SM75 port by lvb-s. [Upstream implementation](https://github.com/vllm-project/vllm/pull/69028) ·
[Community SM75 port](https://github.com/lvb-s/vllm-inkling-sm75)

Upstream updates are re-evaluated within the SM75-specific scope of this fork.
