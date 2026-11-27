<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-definitive-title.jpeg)

Language: English | [简体中文](README.zh-CN.md)

The definitive vLLM inference runtime for dual RTX 2080 Ti and other SM75 GPUs,
including Tesla T10/T40/T4, TITAN RTX, and Quadro RTX 6000/8000.

This hardware-focused fork preserves the SM75-specific source changes, launcher
profiles, and validation evidence needed to reproduce these Turing inference
stacks. It is
based on upstream vLLM; retain both the upstream license and attribution to
`github.com/weicj` when redistributing a derivative.

For usage feedback, feature requests, and community discussion, join the
[Discord community](https://discord.gg/VFqVVySdMS).

![Live single-request throughput demo](docs/assets/vllmspeed_dflash.gif)

Current development version is `v0.3.0.dev0` on `vllm-2080ti-definitive-0.3.x`.
The Docker runtime uses the official [vLLM v0.34.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0).
The native runtime comes from v0.2.2-post3. Each hardware profile guide
records the runtime version used for its benchmark.
[v0.2.2-post3 release](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.2-post3)

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

The supported hardware families also include two 16 GiB Tesla T10 GPUs over
PCIe, with validated TP=2 Qwen 27B routes. The four-GPU Tesla T10 profiles
target TP=4 serving; routes with a noted image-semantic failure remain explicitly
marked as candidates.

## 🧩 Support Status

The native build targets Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15, CUDA 13.0, and PyTorch 2.13. For CUDA 12.8, PyTorch 2.11, older kernels, or GCC 12/13/14, refer to the `0.1.x` line, which is no longer actively maintained.

Supported model routes and their measurements are listed in the corresponding
hardware profile guides.

The launcher supports TP, PP, and mixed TP/PP inference; validated layouts
include two RTX 2080 Ti GPUs with TP=2, two Tesla T10 GPUs with TP=2, and four
Tesla T10 GPUs with TP=4.

## 🧪 Tested Model Checkpoints

Current tested model and weight routes:

| Model route | Weight route | Model card | Recommended use | Profile path |
| --- | --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | High-precision single-request inference | `qwen27b/w8a16` |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Long-context concurrent inference | `qwen27b/w4a16` |
| Qwen3.8 27B | INT4 (W4A16) | [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | 2xT10 MTP3 serving | `qwen27b/w4a16` |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | Fast personal inference | `qwen35b/w8a16` |

## ⚡ Highlights

| Hardware | Weight | Context / KV | 4K Prompt decode | 32K Prompt decode |
| --- | --- | --- | ---: | ---: |
| 2x RTX 2080 Ti | Qwen3.8 27B NVFP4 | 256K / FP8 KV | **220.84 tok/s** | **209.35 tok/s** |
| 4x Tesla T10 | Qwen3.8 27B FP8 | 256K / FP16 KV | **191.89 tok/s** | **189.38 tok/s** |

Both rows are single-request tests using DFlash2 (default K=7) and synthetic text inputs with high speculative-hit rates. Real-task throughput depends on draft acceptance and may not reach these figures.

## 🚀 Build And Launch

1. Run `./build.sh` in the 0.3.x checkout. The default build uses Docker
   and takes the Torch and CUDA stack from the upstream release image. The
   [image lock](PROJECT_RELEASE.env) pins the official linux/amd64 image by
   digest. The build applies the SM75 GDN prefill patch and compiles FlashQLA
   into the image.

   The Docker runtime requires Docker with NVIDIA Container Toolkit and a
   host driver compatible with CUDA 13.0. The existing native builder is
   available with `VLLM_RUNTIME_BACKEND=native ./build.sh` and
   `VLLM_RUNTIME_BACKEND=native ./launcher.sh`.

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

![launcher.sh interactive main menu](docs/assets/launcher-main-menu.png)

For automated deployment, pass the configuration non-interactively:

```bash
MODEL_DIR=/path/to/checkpoint \
PROFILE=2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env \
MODE=fast GPU_DEVICES=1,5 TP_SIZE=2 \
NON_INTERACTIVE=1 ./launcher.sh
```

Use `./launcher.sh --print-config` to preview a route. See the
[non-interactive launch guide](docs/non-interactive-launch.md) for automation.

The Docker launcher uses the same local checkpoint paths, profile files, GPU
order, and service settings. It uses upstream scheduling and fast-mode graph
defaults, with PIECEWISE graphs for speculative decoding in normal mode and
for the dual T10 Qwen TurboQuant MTP text-image route. The server-wide
thinking budget and the 0.2.x prefill barrier are
available through the native runtime.

The Docker runtime is experimental on SM75. The T10 profile guides record
Docker results alongside the 0.2.x measurements.
[2xT10 profile results](profiles/2xT10/README.md#03x-docker-validation) ·
[4xT10 profile results](profiles/4xT10/README.md#03x-docker-validation)

### Inkling 2 Mini on SM75

Inkling 2 Mini support landed on vLLM main after the v0.34.0 release. The
Inkling image backports it to the pinned base.

Build the Inkling image and select it when launching the official checkpoint.
Set the GPU indices and tensor parallel size for your host. This example uses
two PCIe P2P-capable T10s and serves text with an 8,192-token context.

```bash
VLLM_DOCKER_VARIANT=inkling ./build.sh

VLLM_DOCKER_VARIANT=inkling MODEL_DIR=/path/to/Inkling-2-Mini \
GPU_DEVICES=0,1 TP_SIZE=2 MAX_MODEL_LEN=8192 GPU_UTIL=0.94 \
KV_CACHE_DTYPE=fp8 MESSAGE_TYPE=text-only NCCL_P2P_DISABLE=0 \
DISABLE_PREFIX_CACHING=1 DISABLE_LOG_STATS=0 NON_INTERACTIVE=1 ./launcher.sh
```

Set `SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`
on the launcher command to load `mtp.safetensors` from the official model
directory.

The 2×T10 and 4×T10 tests used NCCL PCIe P2P and FP8 KV at
an 8,192-token context limit. Each row reports single-request medians from
three samples of exactly 4,096 input and 128 output tokens with prefix
caching disabled. DSpark uses seven draft tokens.

| GPUs | Prompt | AR prefill / decode (tok/s) | DSpark prefill / decode (tok/s) | Draft acceptance |
| --- | --- | ---: | ---: | ---: |
| 2×T10 | Prose | 3644.31 / 63.67 | 3556.14 / 175.24 | 76.4–81.2% |
| 2×T10 | Code | 3615.24 / 63.40 | 3540.27 / 183.17 | 86.5% |
| 2×T10 | Repetitive | 3673.90 / 62.98 | 3594.91 / 278.34 | 100% |
| 4×T10 | Prose | 4314.14 / 77.71 | 4172.03 / 255.02 | 81.2–86.5% |
| 4×T10 | Code | 4312.15 / 77.37 | 4184.09 / 251.59 | 86.5% |
| 4×T10 | Repetitive | 4192.41 / 76.68 | 4089.48 / 353.55 | 78.6–100% |

The experimental image uses the upstream model implementation and
the SM75 port by lvb-s. [Official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini) ·
[Upstream implementation](https://github.com/vllm-project/vllm/pull/69028) ·
[Community SM75 port](https://github.com/lvb-s/vllm-inkling-sm75)

## 🧭 Profiles

Read the [Profile Guide](profiles/README.md) for the layout and route fields.
Detailed profiles and reference performance are listed for
[2x2080Ti](profiles/2x2080Ti/README.md) and
[2xT10](profiles/2xT10/README.md), and
[4xT10](profiles/4xT10/README.md).

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
- `0.2.x` target: CUDA 13.0, PyTorch 2.13, Python 3.12
- Native build host: Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15

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

The `0.2.x` target is CUDA 13.0 with PyTorch 2.13. For the older CUDA 12.8 /
PyTorch 2.11 stack, refer to the unmaintained `v0.1.x` compatibility line. Keep
the PyTorch CUDA build, toolkit, FlashInfer/FlashQLA build, and selected profile
aligned; they are not interchangeable runtime combinations.

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
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. It
keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and dual-2080-Ti validation notes.

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
