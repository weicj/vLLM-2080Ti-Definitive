# 8xT10 Profiles

Language: English | [简体中文](README.zh-CN.md)

## Qwen3.8 Flash-Next NVFP4 (experimental PP)

These routes target eight Tesla T10 GPUs (`0,2,3,4,6,7,8,9`) and use the
SM75-safe Marlin W4A16 path for ModelOpt NVFP4 weights. They deliberately use
FP16 KV, `MTP_K=0`, `flashqla_legacy`, NCCL collectives, synchronous scheduling, and explicit
`PLE_PLACEMENT=disk` (safetensors mmap/page-cache lookup). `cpu` selects pinned
host memory with UVA, while `gpu` keeps the table resident in device memory.

| Profile | TP/PP | Context | PLE | 4K/128 | 32K/512 |
|---|---:|---:|---|---|---|
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 4x2 | 40K | disk | pending audit | pending audit |
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 2x4 | 40K | disk | pending audit | pending audit |

Run the service first, then execute the auditable runner from the repository
root. It performs one warm-up and three measured samples for each shape and
writes commit, profile, environment, GPU, request-contract, and per-sample
records to a JSON manifest:

```bash
python3 tools/benchmark_qwenflash_audit.py \
  --model-dir /path/to/qwen38-flash-next-nvfp4 \
  --served-name qwen38-flash-next-nvfp4-tp4pp2 \
  --base-url http://127.0.0.1:8000/v1 \
  --python /path/to/runtime/.venv/bin/python \
  --profile profiles/8xT10/qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env \
  --output results/qwenflash/tp4pp2-audit.json
```

The runner rejects incomplete streams and any request whose prompt or
completion token count differs from the requested `4K/128` or `32K/512`
contract. See [the audit protocol](../../docs/qwenflash-audit.md) for the
manifest schema and evidence policy.
