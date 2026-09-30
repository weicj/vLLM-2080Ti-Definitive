#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

test_root=$(mktemp -d)
trap 'rm -rf "$test_root"' EXIT
RUNTIME_ROOT=$test_root
mkdir -p "$RUNTIME_ROOT/.venv/bin"
ln -s "$(command -v python3)" "$RUNTIME_ROOT/.venv/bin/python"
config_path="$test_root/mooncake.json"
printf '%s\n' '{"metadata_server":"127.0.0.1:2379","master_server_address":"127.0.0.1:50051"}' > "$config_path"

parse_launcher_args --kv-store-backend mooncake --mooncake-config-path "$config_path"
[[ "$KV_STORE_BACKEND" == mooncake ]]
[[ "$MOONCAKE_CONFIG_PATH" == "$config_path" ]]
export -n MOONCAKE_CONFIG_PATH
validate_kv_store_config
[[ "$(printenv MOONCAKE_CONFIG_PATH)" == "$config_path" ]]

model_dir="$test_root/model"
mkdir -p "$model_dir"
printf '%s\n' '{"model_type":"qwen3_5","max_position_embeddings":4096}' > "$model_dir/config.json"
preview=$(RUNTIME_ROOT="$test_root" LOG_DIR="$test_root/logs" "$ROOT/launcher.sh" \
  --model-dir "$model_dir" --model-family qwen35 --gpu-devices 0 \
  --tp-size 1 --pp-size 1 --max-model-len 4096 --mode fast \
  --kv-store-backend mooncake --mooncake-config-path "$config_path" --print-config)
grep -Fq 'External KV store:    Mooncake Store (experimental)' <<< "$preview"
grep -Fq "Mooncake config:      $config_path" <<< "$preview"

MODEL_DIR=/tmp/test-model
SERVED_NAME=test-model
PORT=18000
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=512
MAX_NUM_SEQS=1
TP_SIZE=1
MODE=fast
MODEL_FAMILY=qwen
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--kv-transfer-config' <<< "$args"
grep -Fxq -- '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' <<< "$args"

KV_STORE_BACKEND=none
validate_kv_store_config
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
! grep -Fxq -- '--kv-transfer-config' <<< "$args"

KV_STORE_BACKEND=mooncake
MOONCAKE_CONFIG_PATH="$test_root/missing.json"
if validate_kv_store_config 2>/dev/null; then
  echo "missing Mooncake config unexpectedly accepted" >&2
  exit 1
fi
MOONCAKE_CONFIG_PATH=$config_path
printf '%s\n' '{broken' > "$config_path"
if validate_kv_store_config 2>/dev/null; then
  echo "malformed Mooncake config unexpectedly accepted" >&2
  exit 1
fi
printf '%s\n' '{"metadata_server":"127.0.0.1:2379"}' > "$config_path"
if validate_kv_store_config 2>/dev/null; then
  echo "Mooncake config without master unexpectedly accepted" >&2
  exit 1
fi
KV_STORE_BACKEND=unknown
if validate_kv_store_config 2>/dev/null; then
  echo "unknown KV store unexpectedly accepted" >&2
  exit 1
fi

echo "launcher_mooncake_store_ok"
