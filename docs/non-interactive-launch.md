# Non-Interactive Launch

Build the runtime with `./build.sh`, then pass a flat profile path. Profile
files contain route parameters only; checkpoint, GPU topology, port, and mode
remain launcher options.

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
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

## Experimental External KV Store

The launcher can opt into `MooncakeStoreConnector` alongside the local prefix
cache. Mooncake's metadata server, master server, and storage pool must already
be running. Install the CUDA-matched Mooncake transfer-engine package in the
runtime venv and provide a JSON file containing at least `metadata_server` and
`master_server_address`. Set `mode`, `protocol`, `device_name`,
`global_segment_size`, and `local_buffer_size` for your Mooncake deployment;
the connector's embedded mode uses process memory and does not survive a vLLM
restart. Use the Mooncake deployment documentation for the external services.

The external store is a service setting, not a route profile field. For example:

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast --gpu-devices 1,5 --tp-size 2 --pp-size 1 \
  --kv-store-backend mooncake \
  --mooncake-config-path /path/to/mooncake-store.json \
  --print-config
```

Remove `--print-config` to start the service. The launcher checks the JSON
shape and required endpoints during preview, then checks `import mooncake.store` before a real
launch. This is an experimental connector path, not a validated SM75 profile;
verify cache hits, output correctness, and performance before production use.
`MooncakeConnector` for prefill/decode disaggregation is a separate feature.
