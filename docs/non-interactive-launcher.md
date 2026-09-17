# Non-Interactive Launcher

`launcher.sh` supports a complete non-interactive startup path for scripted
deployment, CI checks, and reproducible route validation.

The final precedence is:

`CLI > ENV > PROFILE > default`

`CLI` includes direct flags such as `--max-model-len 262144`, `--set
KEY=VALUE`, and `--unset KEY`.

## Flag Naming

Known launcher keys can be passed directly as `--lower-kebab-case`:

- `MODEL_DIR` -> `--model-dir`
- `PROFILE` -> `--profile`
- `MAX_MODEL_LEN` -> `--max-model-len`
- `MM_LIMIT_JSON` -> `--mm-limit-json`

For advanced environment variables that are not part of the direct launcher
surface, use:

- `--set KEY=VALUE`
- `--unset KEY`

`--unset KEY` clears inherited profile or environment values and then lets the
launcher fall back to lower-priority defaults. If you need to keep an explicit
empty string instead of falling back, use `--set KEY=`.

## Common Examples

Print the resolved launch summary without starting the server:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env
```

Start from a shipped profile and override only a few fields:

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --mode normal \
  --service-scope lan \
  --max-model-len 196608 \
  --gpu-devices 0,1
```

Environment values still work, but CLI wins over them:

```bash
MAX_MODEL_LEN=131072 \
MODE=fast \
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --max-model-len 262144
```

Use `--set` for advanced runtime envs:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_TURBOQUANT_FLASHINFER_BACKEND=fa2 \
  --set CC=/usr/bin/gcc-12
```

Clear a profile-provided field and let the launcher refill the lower-level
default:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --unset max-model-len
```

Force an explicit empty value:

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set REASONING_PARSER=
```

## API Key Authentication

vLLM's OpenAI-compatible server can require a bearer token for `/v1/*` requests.
The launcher passes the key through the environment, so it is not written into
the launch command recorded in `run-logs/` and does not appear in the server
process arguments:

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_API_KEY=<key>
```

When `VLLM_API_KEY` is present, the startup smoke test sends
`Authorization: Bearer $VLLM_API_KEY`, so a launch with authentication enabled
still has to pass the smoke request. With no key in the environment, the smoke
test and the server behave exactly as before.

`VLLM_API_KEY` is a global/advanced runtime env, not a route-profile field, so
do not put it in `profiles/*.env`. Authentication covers `/v1/*` only:
`/health`, `/metrics`, `/version`, `/load`, `/tokenize`, and `/detokenize` stay
open, and `/health` staying open is what the readiness probe relies on. Never
set `VLLM_SERVER_DEV_MODE=1` on an exposed service, because that mounts
unauthenticated control endpoints such as `/sleep`, `/wake_up`,
`/reset_prefix_cache`, and `/collective_rpc`.

## Derived Defaults

Some launcher fields are still normalized after profile loading:

- `MESSAGE_TYPE`, `MM_LIMIT_JSON`, `LANGUAGE_MODEL_ONLY`, and
  `SKIP_MM_PROFILING`
- mode-derived runtime knobs such as `ENFORCE_EAGER`,
  `VLLM_SM75_SPEC_SYNC_MODE`, and
  `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`
- family defaults such as `REASONING_PARSER=qwen3` on validated Qwen3 routes
- Qwen prefix-cache defaults such as `MAMBA_CACHE_MODE=align`

Those defaults only fill missing values. Explicit CLI values keep priority.

## Validation Workflow

Use `--print-config` first when changing scripts or deployment automation. It
prints the final launch summary and exits before starting vLLM, which makes it
the safest way to verify that every override landed on the final config.
