# SM75 Docker image

Language: English | [简体中文](docker-sm75.zh-CN.md)

A ready-to-run vLLM 2080 Ti Definitive Edition container for **RTX 2080 Ti
(SM75)**. It works like the official vLLM image: the entrypoint is
`vllm serve`, so the model and flags go directly after the image name.

## Status

Validated on **two RTX 2080 Ti (2× 22 GiB, tensor parallel 2)**:

- A 27B production-style model served both text and multimodal requests, captured
  CUDA graphs normally, reported a 7.75 GiB KV pool, used about 18.8 GiB per
  card, and produced zero NVRM errors; cold load took about 13 minutes.
- A small text model (`Qwen/Qwen3-0.6B`, FP16) also starts and answers correctly.

Not measured on this hardware: throughput, TTFT/TPOT, KV capacity at long
context, and cache reuse across restarts. Capacity and throughput depend on the
model, KV precision, context length, and topology, so figures from a different
configuration do not transfer. NVLink topology and other models, precisions, or
backends are also unverified.

The host must be able to allocate pinned host memory; see
[Host requirements](#host-requirements).

CI builds the image on a GPU-less machine in about **26 minutes**, pushes it,
pulls it back by digest, and smoke-tests those exact bytes. A green CI run
therefore means the published image is intact — not that GPU inference has been
tested.

## Host requirements

A working NVIDIA driver and GPU passthrough are not enough on their own: the host
must also be able to allocate pinned host memory. If it cannot, no model starts,
and the failure is

    torch.AcceleratorError: CUDA error: OS call failed or operation not supported on this OS

while the kernel log shows the real cause:

    NVRM: Failed to create a DMA mapping!
    NVRM: osIovaMap: failed to map allocation (status = 0x59)

Both come from `nv_dma_map_alloc` handing pages to the kernel DMA API, so the
fault is in the host's IOMMU/DMA layer — not in this image, its flags, or the
model. The official vLLM image fails identically on such a host. Check it without
Docker and without vLLM:

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

`CUresult` `0` is success. Code `304` (`CUDA_ERROR_OPERATING_SYSTEM`) on the
larger sizes is the DMA-mapping failure described above; a different code points
somewhere else, so read it before touching IOMMU settings. If the larger sizes
fail with `304`, put devices into passthrough mode by adding
`iommu.passthrough=1` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`,
then `sudo update-grub && sudo reboot`. Confirm it took effect — the log must
report `Passthrough` — before re-testing:

```bash
journalctl -k -b | grep 'Default domain type'
```

`intel_iommu=pt` is **not** a valid option on current kernels — it only prints
"Unknown option" and silently does nothing. `iommu.passthrough=1` is the option
that works. This is a host-side change: a host whose IOMMU translation domain
cannot map these allocations will fail the same way regardless of which vLLM
image runs on it.

## Quick start

Pull a published image. `weicj/vLLM-2080Ti-Definitive` publishes to
`ghcr.io/weicj/vllm-2080ti-definitive`; a fork publishes under its own owner:

```bash
export IMAGE="ghcr.io/weicj/vllm-2080ti-definitive:main"
docker pull "$IMAGE"
```

### Docker Compose (recommended)

```bash
mkdir vllm-sm75 && cd vllm-sm75
curl -fsSL -o compose.yaml https://raw.githubusercontent.com/weicj/vLLM-2080Ti-Definitive/main/docker/docker-compose.sm75.yml
```

The file is also in the repository at
[`docker/docker-compose.sm75.yml`](../../docker/docker-compose.sm75.yml).

Configure it before starting. `MODELS_DIR` is the host directory that holds your
models; `MODEL_NAME` is the directory name under it. Both are required — compose
stops with an error if either is missing, so you never end up serving a model you
did not choose.

```bash
cat > .env <<'EOF'
MODELS_DIR=/path/to/models
MODEL_NAME=Qwen3-0.6B
EOF
docker compose up -d
```

Optional: `IMAGE`, `MODEL_ALIAS`, `TP_SIZE`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `MAX_NUM_SEQS`. Add any further vLLM flag —
`--api-key`, `--kv-cache-dtype`, `--enable-prefix-caching`, and so on — to
`command:` in the compose file; the image entrypoint is already `vllm serve`.
The first start loads the weights and may compile kernels, so give it time:

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

Pass the model and flags directly — do not repeat `vllm serve`, the entrypoint
already adds it.

Check that it is serving, from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

`--ipc=host` is what the official image documents, and multi-GPU tensor
parallel needs it (or a large enough `--shm-size`).

On a **WSL2 host**, vLLM disables pinned memory by default and startup aborts
with `RuntimeError: UVA is not available`. Add `-e VLLM_WSL2_ENABLE_PIN_MEMORY=1`
there; native Linux does not need it.

Both the compose file and the `docker run` examples publish to `127.0.0.1`. The
server has no authentication unless you configure it, so add `--api-key` before
exposing it beyond the local machine.

## Two GPUs

Same command with `--tensor-parallel-size 2`:

```bash
docker run --rm --name vllm-sm75-2gpu --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  "$IMAGE" /models/Qwen3-0.6B \
  --dtype half --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.9
```

Confirm both cards are in use, and run the same `/health`, `/v1/models`, and
completion checks as above:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

With Compose, add `--tensor-parallel-size 2` to `command:` in the compose file
instead of passing it here.

Start with a small model to prove the pipeline works before trying one that
really needs both cards. On the exact machine, record whether the cards are
joined by an NVLink bridge or only PCIe, the cold-start time, the first-use JIT
compile time, the second-start time with the same cache volumes, per-device
memory, and whether the output is still correct at
`--tensor-parallel-size 2`. Two-GPU serving is now validated (see Status), but the
numbers above are per-machine and still say nothing about capacity or throughput
for other models, precisions, or contexts.

## What the image contains

- **Linux amd64 only**, CUDA 13.0.3 devel, Ubuntu 24.04, Python 3.12 in
  `/opt/venv`. This is newer than the project's native host target
  (Ubuntu 26.04+, kernel 7+, GCC 15) and is a separate, still-narrow route.
- A wheel **built from this repository's source** — not an upstream prebuilt
  vLLM wheel — plus the fork's Torch E8M0 patch.
- FlashQLA as pinned
  [`weicj/FlashQLA-SM70-SM75` source](https://github.com/weicj/FlashQLA-SM70-SM75/tree/3ab27d77d8ca01d7a4718903b726add1a8886c0e)
  with this repository's SM75 patches, at `/opt/FlashQLA`, reachable on
  `PYTHONPATH` and through `FLASHQLA_ROOT` / `FLASHQLA_DIR`. It is source
  integration, not a separately installed FlashQLA distribution with its older
  dependency pins.
- The CUDA toolkit, a C++ compiler, Ninja, and headers, because FlashQLA and
  FlashInfer compile kernels on **first GPU use**. Nothing is AOT-baked, and no
  GPU code runs at image build time.
- `/opt/vllm-tools/` with the release metadata, `pyproject.toml`, and the Torch
  E8M0 checker, for inspection.

The host still needs a working NVIDIA driver and GPU passthrough
(NVIDIA Container Toolkit); the container does not replace the driver. ARM/QEMU
and other GPU architectures are not supported by this route.

Keep the credit and licenses when redistributing an image: [upstream
vLLM](https://github.com/vllm-project/vllm) (Apache-2.0), **vLLM 2080 Ti
Definitive Edition**, author [`github.com/weicj`](https://github.com/weicj),
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA), its
[SM70/SM75 adaptation](https://github.com/weicj/FlashQLA-SM70-SM75), and the
other bundled third-party components.

## Build it yourself

Usually you do not need this — CI publishes the image. To build locally:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.sm75 --target vllm-openai \
  --build-arg MAX_JOBS=2 --build-arg NVCC_THREADS=1 \
  --load -t vllm-sm75:local .
export IMAGE=vllm-sm75:local
```

This compiles CUDA kernels and the Rust frontend from source, so it needs
substantial RAM, disk, and time. CI keeps compiler concurrency low and allows
about 350 minutes; neither is a guarantee for your machine. The build stage must
also be able to run `git` against the pinned FlashQLA repository.

## Tags and publication

Images are published to GHCR under the lowercased repository name
(`ghcr.io/<owner>/vllm-2080ti-definitive`):

| Event | Tags published after checks pass |
| --- | --- |
| Pull request, or push to a non-default branch | none — build and smoke only |
| Push to the default branch | `main` and `sha-<commit>`; `latest` is not moved |
| Published GitHub Release | the release tag, e.g. `v0.2.2-post3`; `latest` only for a non-prerelease |
| Manual run | build-only by default; publishing creates a unique `test-…` tag |

Stability comes from GitHub's own prerelease flag, so `post3` is not
misinterpreted as a prerelease by generic semver rules. Pushing a Git tag alone
publishes nothing. Pull requests never get package-write credentials and are
never sent to a personal GPU runner.

GHCR packages can start out private: you may need to authenticate, or to set the
package visibility explicitly. Check the OCI `source` and `revision` labels, and
prefer a digest over a tag for reproducible deployments.

## Verifying an image

These checks never start CUDA or load native kernels, so they also work on a
machine with no GPU:

```bash
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py
docker run --rm --entrypoint uv "$IMAGE" pip check --python /opt/venv/bin/python
```

The first command checks the wheel version, Torch/CUDA versions, `import vllm`,
the Torch E8M0 patch, the FlashQLA source and patches, the expected paths, and
the JIT tools. The second checks that the dependency set is consistent.

`vllm serve --help` cannot be used as-is here: with no GPU it fails while
detecting the device. The separate CLI check below runs the real installed
console script with only that missing device metadata supplied in an isolated
process. It verifies CLI imports and argument parsing — not GPU startup:

```bash
docker image inspect --format '{{json .Config.Entrypoint}}' "$IMAGE"
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py --cli-help
```

The expected entrypoint is `["vllm","serve"]`. On a real GPU host you can also
check the entrypoint untouched with `docker run --rm --gpus all "$IMAGE" --help`.

To run actual kernels, pass `--gpu` explicitly. It fails loudly on a missing or
non-SM75 GPU instead of skipping, and only tests device 0:

```bash
docker run --rm --gpus all --ipc=host \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  --entrypoint python "$IMAGE" /opt/vllm-tools/smoke_docker_sm75.py --gpu
```

This exercises a vLLM core SiLU op, FlashInfer's CUDA-core decode against an
FP32 reference, and FlashQLA's JIT `gdn_forward` / `gdn_forward_varlen`
(including native FP16 Q/K/V versus FP32, and packed versus separate sequences).
It uses tiny tensors: it is not a benchmark, a model download, a backend
qualification, or a tensor-parallel test.

After the first successful JIT, run the same command again in a fresh container
with the same volumes to check cache reuse — and record the two runs separately,
because a passing rerun alone does not prove nothing recompiled.
`TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` is the persistent FlashQLA
cache. Use separate caches for incompatible image or toolchain updates, and
never mount a host virtualenv over `/opt/venv`.

## Serving a real model

Serving your own weights is the normal case: mount the directory read-only and
pass the path inside the container, as in Quick start. `--served-model-name`
sets the ID that clients use.

To let vLLM download weights from the Hugging Face Hub instead, mount a cache
directory so they are not fetched again on every recreate, and pass `HF_TOKEN` if
the repository is gated — never bake the token into an image:

```bash
docker run --rm --name vllm-sm75-hub --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN "$IMAGE" Qwen/Qwen3-0.6B \
  --dtype half --max-model-len 2048 \
  --host 0.0.0.0 --port 8000
```

Then check health, the served model ID, and a real non-empty completion — not
just the HTTP status:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
curl --fail --silent --show-error http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

`Qwen/Qwen3-0.6B` in these examples is a smoke-sized model chosen to load fast,
not a recommended deployment profile.

For Qwen3.5-family GDN routes that use the fork's FlashQLA backend, pass
`--gdn-prefill-backend flashqla_legacy` explicitly (not
`--mamba-backend flashqla_legacy`), together with `--dtype half` and the rest of
the route settings. The Qwen3-0.6B smoke does not exercise GDN.

The image does not run `launcher.sh` and does not apply profiles: GPU indices,
tensor parallel size, context, MTP, reasoning, and chat-template settings are
yours to pass. To reproduce a native deployment, read the intended launcher's
effective configuration with `launcher.sh --print-config` and translate it into
flags and environment variables; the
[hardware profile guides](../../profiles/README.md) describe the routes. Check
the exact model, precision, MTP setting, context, topology, and benchmark method
before reusing any native-host performance number.