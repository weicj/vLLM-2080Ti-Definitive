# DFlash Experiments

This document records the historical DFlash experiment matrix and the promoted
DFlash2 route for the 27B Qwen3.8 stack in this repository.

Status:

- The older Qwen3.6 DFlash2/DFlash3 matrix below remains experimental.
- The Qwen3.8 NVFP4 DFlash2 route is now promoted in the normal profile
  catalog after validation on the target dual-2080-Ti runtime.

The promoted profile is:

- `qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env`

It uses `nvidia/Qwen3.8-27B-NVFP4` with `incoai/Qwen3.8-27B-DFlash2`,
TurboQuant K8V4 KV, K=7, TRITON_ATTN draft attention, PIECEWISE CUDA Graphs,
and the SM75 causal target-verification path in B=8 chunks. Three fixed 4K/128
high-acceptance runs measured 155.16, 166.57, and 166.76 decode tok/s. The
route is formally 256K text-only and passed correctness, near-full-context,
and multi-thousand-token output-stage stress tests.

The current experimental profiles are:

- `qwen27b/experimental/fp8/fp16kv-8K-dflash3-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash2-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash2-drafttp1-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash3-drafttp1-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-128K-dflash3-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash2-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash2-drafttp1-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash3-drafttp1-text-only.env`
- `qwen27b/experimental/int4/fp16kv-256K-dflash3-text-only.env`

The 8K profiles above intentionally cover the current short-route sweep matrix:

- `dflash2` with draft TP `auto`
- `dflash2` with draft TP `1`
- `dflash3` with draft TP `auto`
- `dflash3` with draft TP `1`

These experimental startup profiles use the launcher shortcut form
(`SPECULATIVE_METHOD=dflash` plus `SPECULATIVE_TOKENS`). Do not mix DFlash
shortcut profiles with `MTP_K` or a separate `SPECULATIVE_CONFIG`.

These historical profiles stay outside the shipped `normal/` and `fast/`
catalog because they do not have the same promotion evidence:

- quality smoke passing,
- valid service startup on the dual-2080-Ti runtime,
- and synthetic decode speed beating the generated MTP3 baseline on the short
  speed route.

## Prerequisites

- A validated dual-2080-Ti runtime with tensor parallel size `2`.
- A target 27B FP8 model directory.
- A target 27B INT4 model directory.
- A matched DFlash draft model, typically `z-lab/Qwen3.6-27B-DFlash`.

If the draft model is given as a Hugging Face repo ID instead of a local path,
the launcher auto-probes the official endpoint and the mirror endpoint at real
launch time, then picks the reachable faster route. To pin the route manually,
set `HF_ENDPOINT` directly or `HF_DOWNLOAD_ROUTE_MODE=official|mirror`.

Before running a real comparison, dry-run the launcher:

```bash
bash launcher.sh --print-config \
  --model-dir /path/to/target-27b \
  --profile qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env \
  --gpu-devices 0,1 \
  --tp-size 2
```

The summary should show `Spec decode: dflash/3 (...)`. If the draft model is a
repo ID, `DFlash draft fetch` will show the planned route policy; actual
endpoint selection happens only when the service really launches.

## Single Profile Comparison

Use the service-side compare helper to compare a DFlash profile against a
generated MTP3 baseline derived from the same base profile:

```bash
bash tools/compare_dflash_service.sh \
  --case qwen27b-int4-8k \
  --model-dir /path/to/int4-27b \
  --profile qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env \
  --draft-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --out-dir /tmp/qwen27b-int4-dflash \
  --require-dflash-beats-mtp
```

This helper:

- generates a temporary MTP3 route from the same base profile,
- runs `launcher.sh --print-config`,
- starts the service through `launcher.sh --non-interactive`,
- runs a deterministic `PROFILE_OK` quality smoke and requires the DFlash reply
  to match the MTP baseline reply,
- runs warmup + synthetic completion requests,
- writes `*-compare-service.json`.

## Batch Evaluation

Use the batch helper to run the current default four-case experimental set:

```bash
bash tools/evaluate_dflash_profiles.sh \
  --fp8-model-dir /path/to/fp8-27b \
  --int4-model-dir /path/to/int4-27b \
  --dflash-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2
```

The default batch case set is:

- FP8 8K: `fast`, speed gate required.
- INT4 8K: `fast`, speed gate required.
- FP8 128K: `normal`, quality observation.
- INT4 256K: `normal`, quality observation.

The batch helper also accepts DFlash tuning passthrough flags:

- `--speculative-tokens`
- `--draft-tp-size`
- `--draft-max-model-len`
- `--draft-attention-backend`
- `--disable-padded-drafter-batch`

Outputs:

- `cases.tsv`
- `summary.tsv`
- `verdict.txt`
- one subdirectory per case with compare JSON and logs
- one `spec_decode_metrics.json` per variant when the server exposes Prometheus
  speculative-decoding counters

## Speed Sweep

Use the speed-sweep helper to scan short 8K FP8/INT4 routes across candidate
token counts and draft TP settings:

```bash
bash tools/evaluate_dflash_speed_sweep.sh \
  --fp8-model-dir /path/to/fp8-27b \
  --int4-model-dir /path/to/int4-27b \
  --dflash-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --speculative-tokens-list 2,3 \
  --draft-tp-size-list auto,1
```

Outputs:

- `combined-summary.tsv`: one row per weight x tuning candidate
- `best.tsv`: best passing candidate per weight by decode ratio
- one subdirectory per candidate with the full per-run batch result

After choosing a winning candidate, materialize it into a concrete startup
profile:

```bash
bash tools/materialize_dflash_profile.sh \
  --base-profile qwen27b/experimental/fp8/fp16kv-8K-dflash3-text-only.env \
  --output /tmp/fp16kv-8K-dflash2-drafttp1-text-only.env \
  --speculative-tokens 2 \
  --draft-tp-size 1
```

Or materialize the full `best.tsv` result set in one step:

```bash
python3 tools/materialize_dflash_profiles_from_best.py \
  --best-tsv /path/to/dflash_speed_sweep/best.tsv \
  --output-root /tmp/dflash-materialized
```

## Remote Target Execution

If your current workspace is not the target runtime, use the remote wrapper to
sync the DFlash experiment files into an already prepared runtime tree and run
the batch there:

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --probe-only \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash
```

If the probe reports valid GPU inventory, runtime Python, and model paths, run
the actual batch:

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2
```

This wrapper:

- can run `--probe-only` to check remote GPU visibility, runtime paths, model
  paths, and Python-side DFlash support before syncing or benchmarking,
- optionally separates the SSH login user (`--ssh-user`) from the runtime owner
  user (`--remote-user`),
- syncs `profiles/`, `tools/`, `launcher.sh`, `build.sh`, `VERSION`, and the
  current local `vllm/` tree into the remote runtime,
- runs `tools/evaluate_dflash_profiles.sh` on the remote runtime as the target
  runtime user,
- copies the result directory back to the local workspace.

The remote wrapper also accepts the same DFlash tuning passthrough flags as the
batch helper.

To run the short speed sweep directly on the remote runtime, switch the runner:

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --runner speed-sweep \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --speculative-tokens-list 2,3 \
  --draft-tp-size-list auto,1
```

Interpretation:

- `quality_gate_ok=True` means both the generated MTP3 route and the DFlash
  route returned HTTP 200, passed the `PROFILE_OK` smoke, and matched on the
  deterministic quality reply.
- `quality_text_equal_stripped=True` means the two deterministic quality replies
  matched after trimming surrounding whitespace.
- `request_gate_ok=True` means the measured synthetic requests all completed.
- `dflash_beats_mtp=True` is only a hard requirement on the short 8K speed
  routes by default.
- `mtp_acceptance_rate` and `dflash_acceptance_rate` in `summary.tsv` help
  explain why one route wins or loses on decode speed.

Long-context DFlash routes should not be promoted from `experimental` until
they also have route-specific capacity evidence and quality evidence on the
target runtime.
