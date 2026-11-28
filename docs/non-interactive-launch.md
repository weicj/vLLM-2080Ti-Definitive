# Non-Interactive Launch

Build the runtime with `./build.sh`, then pass a flat profile path. Profile
files contain route parameters only; checkpoint, GPU topology, port, and mode
remain launcher options.

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-INT4 \
  --profile 2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
  --mode fast \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` is optional and defaults to `fast`. A profile without `MODE` keeps
the launcher selection; an explicit profile `MODE=normal` or `MODE=fast` may
override it. There are no `fast/` or `normal/` profile directories.

Useful options include `--model-dir`, `--speculative-model`, `--profile`,
`--mode`, `--gpu-devices`, `--tp-size`, `--pp-size`, `--port`,
`--start-timeout`, and `--print-config`. Use `--set KEY=VALUE` for advanced
launcher/runtime settings. Do not add Prefix Cache, Mamba cache, GPU, port, or
model-path fields to a profile; the validator rejects them.

The profile library is a validation matrix, not a promise that every filename
fits every machine. Capacity and performance are promoted only after the
external audit records startup, 4K/128, 32K/512, concurrency, and image
correctness evidence.

For browser clients hosted on a different origin, set `API_ALLOWED_ORIGINS` to a JSON list of origins when starting the service. For example, add `--set 'API_ALLOWED_ORIGINS=["https://weicj.github.io"]'` for LLM Speed Chat. Use the scheme, host and optional port, without a URL path. If the browser prompts for Local Network Access, grant it to the client site.

## Experimental SSD Prefix Cache

The launcher can opt into vLLM's built-in `OffloadingConnector` with a CPU
staging tier and a filesystem tier. This stores completed prefix KV blocks on
disk without a separate cache daemon. Keep the directory on persistent storage
and use the same model path, KV precision, block layout, and parallel settings
after restarting. The CPU staging allocation uses host RAM; it is not GPU KV
capacity. On each launch, the launcher hashes every file in the target and
local draft checkpoints before loading the model. The content digest selects
a separate disk namespace, so replacing weights at the same path cannot load
old KV. This adds one full checkpoint read to startup; old namespaces must be
removed manually to reclaim space. Clear the cache after changing to an
incompatible runtime. Disk caching requires local checkpoint directories;
do not modify checkpoint files while the launcher starts or the service runs.

The filesystem tier does not evict old files or enforce a disk quota. Use a
dedicated volume with sufficient free space, monitor its usage, and clear stale
cache data while the service is stopped. `KV_DISK_CPU_BYTES` limits only host
RAM staging. The launcher fixes `PYTHONHASHSEED=0` when disk caching is enabled
and rejects another value, so block hashes stay stable across restarts.

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,1,2,3 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/path/to/ssd-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

Remove `--print-config` to start the service. This option requires prefix
caching and sets a stable `PYTHONHASHSEED`. Before stopping the service, wait
for `vllm:kv_offload_tiering_active_cascade_jobs` to return to zero. Compare
`vllm:external_prefix_cache_hits_total` and the filesystem tier's
`vllm:kv_offload_tiering_chunk_hits_total` and
`vllm:kv_offload_tiering_read_bytes_total` after restarting.
