<!-- markdownlint-disable MD001 MD041 -->
# vLLM 2080 Ti Definitive Edition

![vLLM 2080 Ti Definitive Edition cover](docs/assets/vllm-2080ti-cover.jpg)

Hardware-focused vLLM fork for dual RTX 2080 Ti 22 GB / SM75 serving. This is the `vllm-2080ti-definitive-0.2.x` maintenance branch: the public `0.2.x` prerelease line rebased on upstream vLLM `v0.27.1`, CUDA 13.0, and PyTorch 2.13. It is not the stable production default; the maintained `0.1.x` line remains the project's primary stable release line for CUDA 12.8 and PyTorch 2.11.

This project preserves the SM75-specific source changes, launcher profiles,
and benchmark evidence needed to reproduce the dual-2080-Ti TP=2 stack. It is
based on upstream vLLM; retain both the upstream license and attribution to
`github.com/weicj` when redistributing a derivative.

Language: English | [Simplified Chinese](README.zh-CN.md)

![Live single-request throughput demo](docs/assets/vllmspeed.gif)

Fork release: `0.2.1-pre3`
Base vLLM: `0.27.1`

Branch: [`vllm-2080ti-definitive-0.2.x`](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/vllm-2080ti-definitive-0.2.x)
Prerelease snapshot: [v0.2.1-pre3](https://github.com/weicj/vLLM-2080Ti-Definitive/releases/tag/v0.2.1-pre3)
Changes since pre2: [CHANGELOG.md](CHANGELOG.md)

## Why RTX 2080 Ti For LLM Inference?

The project is built around a practical cost/performance premise: two 22 GB
RTX 2080 Ti cards joined by NVLink provide 44 GB of VRAM, substantial memory
bandwidth, and 136 Turing SMs. With an SM75-aware vLLM route, that is enough
for serious local 27B and 35B-class serving rather than only small-model use.

| Metric | 2x RTX 2080 Ti 22 GB + NVLink | RTX 3090 Ti 24 GB baseline | Ratio |
| --- | ---: | ---: | ---: |
| Physical CUDA cores | 8,704 | 5,376 | 1.62x |
| SM count | 136 | 84 | 1.62x |
| Physical Tensor Cores | 1,088 | 336 | 3.24x |
| Dense FP16 matrix throughput | 228 TFLOPS | 160 TFLOPS | 1.43x |
| Total memory bandwidth | 1,232 GB/s | 1,008 GB/s | 1.22x |
| Total VRAM | 44 GB | 24 GB | 1.83x |

The fork turns those hardware properties into a usable serving stack through
Marlin, FlashInfer/FlashQLA, TurboQuant/INT8 KV, MTP, and CUDA Graph support.

## Status

The `0.2.x` target is Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15, CUDA 13.0, and PyTorch 2.13. The maintained `0.1.x` line remains the compatibility route for CUDA 12.8, PyTorch 2.11, older kernels, and GCC 12/13/14.

The dual-2080-Ti CUDA Graph validation is recorded in
[the migration report](docs/2080ti-0.2.1-pre-validation.md). It includes the
exact host selection rule, build gate, correctness regressions, and 4K/128
benchmark method. Do not treat a checkpoint that merely loads as a promoted
deployment route.

The disposition of the previously merged SM75 PRs is recorded in [the 0.2.x PR migration audit](docs/0.2.x-pr-migration-audit.md).

The launcher supports selecting tensor parallelism (`TP_SIZE`) and pipeline
parallelism (`PP_SIZE`), including mixed TP/PP inference layouts when the
visible GPU count matches the requested topology. The primary validated
deployment remains dual RTX 2080 Ti with TP=2 and PP=1; other layouts are
available for engineering tests and require separate validation.

## Tested Model Checkpoints

Current model and weight routes. Individual serving presets and measurements
are listed in the [Profile Guide](profiles/README.md).

| Model route | Weight route | Model card | Recommended use |
| --- | --- | --- | --- |
| Qwen3.8 27B | FP8 | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | High-precision single-request inference |
| Qwen3.8 27B | NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | Long-context concurrent inference |
| Qwen3.x 35B | FP8 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | Fast personal inference |

## Build And Launch

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

Use `./launcher.sh` for interactive setup or `./launcher.sh --print-config` to
preview a route. See the [Profile Guide](profiles/README.md) for available
profiles.

## Profiles

Start with [the Profile Guide](profiles/README.md). Profiles use the layout
`profiles/<model>/<weight>/<mode>/<route>.env`; for example,
`qwen27b/w8a16/normal/fp16kv-128K-mtp3-text-only.env`,
`qwen35b/w8a16/normal/fp16kv-256K-nomtp-text-only.env`, and
`qwen35b/w8a16/normal/fp16kv-136K-nomtp-text-image.env`.

Available modes:

- `normal`: stable daily deployment mode.
- `fast`: higher-performance mode, to be used only with a validated route.
- `aggressive`: highest-performance mode with increased quality risk.
- `safe`: conservative fallback for troubleshooting and compatibility.

The profile selects route parameters. The launcher owns GPU selection, port,
model path, chat template, and reasoning defaults; experimental profiles may
also carry a validated TP/PP layout.

## MTP And KV Precision

Use a shipped profile before hand-tuning MTP and KV settings. Choose KV by
intent: FP16/default KV for output quality, INT8 KV for balanced long-context
service, and TurboQuant K8V4 for compressed fast routes. MTP gains depend on
acceptance rate, so a synthetic peak must be checked against real output and
the profile's quality probe.

For the current migration, use the exact method and measurements in
[the validation report](docs/2080ti-0.2.1-pre-validation.md), especially for
TurboQuant and MTP3. Historical profile capacities are not cu130 evidence.
INT6/AutoRound checkpoints require `humming-kernels[cu13]==0.1.13`, which is
the version pinned by this branch; the full Minachist route remains unverified.

## Hardware Target

- Two RTX 2080 Ti 22 GB GPUs connected by NVLink
- NVIDIA Turing / SM75, tensor parallel size 2
- `0.2.1-pre3` target: CUDA 13.0, PyTorch 2.13, Python 3.12
- Target host: Ubuntu 26.04 or later, Linux kernel 7 or later, GCC/G++ 15

Other Turing cards need independent validation for VRAM capacity, PCIe/NVLink
topology, model head dimensions, KV-cache dtype, and CUDA Graph behavior.

## Hardware Q&A

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

The `0.2.1-pre3` target is CUDA 13.0 with PyTorch 2.13. The older CUDA 12.8 /
PyTorch 2.11 stack remains a separate `v0.1.x` compatibility line. Keep the
PyTorch CUDA build, toolkit, FlashInfer/FlashQLA build, and selected profile
aligned; they are not interchangeable runtime combinations.

**What other hardware risks matter?**

Cooling, stable power delivery, and enough SSD capacity for weights and compile
caches. Thermal throttling can look like a software performance regression,
particularly during long prefill and repeated CUDA Graph/AOT compilation.

## Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual-2080-Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime.

## Credits And Upstream Projects

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
