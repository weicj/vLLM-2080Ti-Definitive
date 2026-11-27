#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

dflash_profile=$(mktemp)
trap 'rm -f "$dflash_profile"' EXIT
printf '%s\n' \
  'MODEL_FAMILY=qwen35moe' \
  'MODEL_VARIANT=nvfp4' \
  'KV_CACHE_DTYPE=turboquant_4bit_nc' \
  'MAX_NUM_SEQS=2' \
  'SPECULATIVE_METHOD=dflash' \
  'SPECULATIVE_TOKENS=7' > "$dflash_profile"

unset LONG_PREFILL_TOKEN_THRESHOLD TP_SIZE
apply_profile_overrides "$dflash_profile"
TP_SIZE=2
apply_speculative_runtime_defaults
[[ "$LONG_PREFILL_TOKEN_THRESHOLD" == "2560" ]]

MODEL_DIR=/tmp/test-model
SERVED_NAME=test-model
PORT=18000
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=512
MAX_NUM_SEQS=3
TP_SIZE=2
MODE=fast
MODEL_FAMILY=qwen
SPECULATIVE_CONFIG='{"method":"dflash","model":"draft","num_speculative_tokens":7}'
LONG_PREFILL_TOKEN_THRESHOLD=256
VLLM_ARGS=()

build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--long-prefill-token-threshold' <<<"$args"
grep -Fxq -- '256' <<<"$args"
! grep -Fxq -- '--compilation-config' <<<"$args"

MODE=normal
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '{"cudagraph_mode":"PIECEWISE"}' <<<"$args"

capacity_log=$(mktemp)
trap 'rm -f "$dflash_profile" "$capacity_log"' EXIT
printf '%s\n' 'GPU KV cache size: 357,194 tokens, Maximum concurrency: 2.03x' > "$capacity_log"
MAX_MODEL_LEN=176128
MAX_NUM_SEQS=2
validate_configured_kv_concurrency "$capacity_log" >/dev/null
MAX_MODEL_LEN=180224
if validate_configured_kv_concurrency "$capacity_log" >/dev/null 2>&1; then
  echo "expected insufficient KV concurrency to fail" >&2
  exit 1
fi

echo "launcher_cudagraph_capture_sizes_ok"
