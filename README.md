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

Current development version is `v0.3.0.dev0` on `vllm-2080ti-definitive-0.3.x`.
**The 0.3.x branch runs on Docker.** `./build.sh` applies the SM75 patches
and adds the FlashQLA extension to the official vLLM v0.34.0 image.
`./launcher.sh` starts the resulting image with your model weights mounted
from the host. The container runs a single copy of vLLM, installed in the
image. [Official vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)

This repository maintains the SM75 patches, FlashQLA extension, launcher,
and hardware profiles. Native builds are available in the [0.2.x release](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.2-post3).

Release history: [CHANGELOG.md](CHANGELOG.md)

## 💡 Why RTX 2080 Ti For LLM Inference?

The project is built around a practical cost/performance premise: two 22 GB
RTX 2080 Ti cards joined by NVLink provide 44 GB of VRAM, substantial memory
bandwidth, and 136 Turing SMs. With an SM75-aware vLLM route, that is enough
for serious local 27B and 35B-class serving rather than only small-model use.

| Metric | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB baseline | Ratio |
| --- | ---: | ---: | ---: |
| Dedicated FP32 datapaths | 8,704 | 5,376 | 1.62x |
| SM count | 136 | 84 | 1.62x |
| Tensor Cores | 1,088 | 336 | 3.24x |
| Dense FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| Total memory bandwidth | 1,232 GB/s | 1,008 GB/s | 1.22x |
| Total VRAM | 44 GB | 24 GB | 1.83x |

The fork turns those hardware properties into a usable serving stack through
Marlin, FlashInfer/FlashQLA, TurboQuant/INT8 KV, MTP/DFlash2, and CUDA Graph support.

## 🧩 Support Status

The launcher supports TP, PP, and mixed TP/PP inference. The 0.3.x Docker
runtime was validated with Qwen and Inkling on two T10s (TP=2) and four
T10s (TP=4).

## 🧪 Validated Models And Profiles

| Hardware | Model | Weights | Profile guide |
| --- | --- | --- | --- |
| 2×T10 | Qwen3.8-27B | INT4 (W4A16) | [Qwen3.8-27B-INT4](profiles/2xT10/README.md#qwen38-27b-int4) |
| 4×T10 | Qwen3.8-27B | FP8 (W8A16), NVFP4 (W4A16) | [FP8](profiles/4xT10/README.md#qwen38-27b-fp8) · [NVFP4](profiles/4xT10/README.md#qwen38-27b-nvfp4) |
| 2×T10, 4×T10 | Inkling 2 Mini | Official checkpoint | [2×T10 AR and DSpark/7](profiles/2xT10/README.md#inkling-2-mini) · [4×T10 AR and DSpark/7](profiles/4xT10/README.md#inkling-2-mini) |

## 🚀 Build And Launch

1. Run `./build.sh` in the 0.3.x checkout. The Docker build takes the Torch
   and CUDA stack from the upstream release image. The
   [image lock](PROJECT_RELEASE.env) pins the official linux/amd64 image by
   digest. The build applies the SM75 patches, compiles FlashQLA, and includes
   Inkling 2 Mini support in the same image used for Qwen.

   The Docker runtime requires Docker with NVIDIA Container Toolkit and a
   host driver compatible with CUDA 13.0. The image provides the Python,
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

For automated deployment, pass the configuration non-interactively:

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-INT4 \
PROFILE=2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=0,1 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

Use `./launcher.sh --print-config` to preview a route. See the
[non-interactive launch guide](docs/non-interactive-launch.md) for automation.

The Docker launcher uses the same local checkpoint paths, profile files, GPU
order, and service settings. It uses upstream scheduling and fast-mode graph
defaults, with PIECEWISE graphs for speculative decoding in normal mode and
for the dual T10 Qwen TurboQuant MTP text-image route.

### Inkling 2 Mini on SM75

Run `./build.sh` after updating to include Inkling 2 Mini support in the local
image. Then select the model checkpoint and profile. This example uses two
T10s with PCIe P2P.

```bash
MODEL_DIR=/path/to/Inkling-2-Mini \
PROFILE=2xT10/inkling-2-mini/official/dspark7-fp8kv-1x8K-text-only.env \
GPU_DEVICES=0,1 TP_SIZE=2 NCCL_P2P_DISABLE=0 DISABLE_PREFIX_CACHING=1 \
NON_INTERACTIVE=1 ./launcher.sh
```

For pure AR, select `nomtp-fp8kv-1x8K-text-only.env`. For four cards, select
the corresponding `4xT10` profile, set `TP_SIZE=4`, and select four GPUs with
PCIe P2P.

Inkling support backports the upstream model implementation to vLLM v0.34.0
and uses the SM75 port by lvb-s. [Official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) ·
[Upstream implementation](https://github.com/vllm-project/vllm/pull/69028) ·
[Community SM75 port](https://github.com/lvb-s/vllm-inkling-sm75)

## 🧭 Profiles

The [Profile Guide](profiles/README.md) explains the layout and route fields.
The [2x2080Ti guide](profiles/2x2080Ti/README.md) lists the dual RTX 2080 Ti profiles pending
Docker validation.

Profiles use the flat layout `profiles/<hardware>/<model>/<weight>/<route>.env`.
The launcher selects the startup mode and defaults to `MODE=fast`; the mode can
also be set explicitly by the launcher or profile.

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

Upstream updates are re-evaluated within the SM75-specific scope of this fork.
