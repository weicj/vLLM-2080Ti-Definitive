#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=../../launcher.sh
source "$ROOT/launcher.sh"

assert_eq() {
  local expected=$1
  local actual=$2
  local label=$3
  [[ "$actual" == "$expected" ]] || {
    printf '%s: expected %q, got %q\n' "$label" "$expected" "$actual" >&2
    return 1
  }
}

unset MODE
normalize_mode
assert_eq fast "$MODE" "default launch mode"
init_config_registry
[[ -n "${CONFIG_KNOWN_KEYS[MM_LIMIT_JSON]+x}" ]] || {
  echo "MM_LIMIT_JSON missing from config registry" >&2
  exit 1
}

reset_message_type_test_state() {
  unset MESSAGE_TYPE MM_LIMIT_JSON MM_IMAGE_LIMIT LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING
  CONFIG_OVERRIDE_SOURCE=()
  CONFIG_OVERRIDE_UNSET=()
}

reset_message_type_test_state
MESSAGE_TYPE=text-only
normalize_message_type_defaults
assert_eq text-only "$MESSAGE_TYPE" "text-only message type"
assert_eq "" "${MM_LIMIT_JSON:-}" "text-only multimodal limit"
assert_eq "" "${MM_IMAGE_LIMIT:-}" "text-only image limit"

reset_message_type_test_state
MESSAGE_TYPE=text+image
normalize_message_type_defaults
assert_eq 64 "$MM_IMAGE_LIMIT" "default image limit"
assert_eq '{"image":64,"video":0,"audio":0}' "$MM_LIMIT_JSON" "default multimodal limit"

reset_message_type_test_state
MESSAGE_TYPE=text+image
MM_IMAGE_LIMIT=12
normalize_message_type_defaults
assert_eq '{"image":12,"video":0,"audio":0}' "$MM_LIMIT_JSON" "custom multimodal limit"

reset_message_type_test_state
MESSAGE_TYPE=text+image
MM_LIMIT_JSON='{"image":3,"video":0,"audio":0}'
MM_IMAGE_LIMIT=invalid
CONFIG_OVERRIDE_SOURCE[MM_LIMIT_JSON]=cli
normalize_message_type_defaults
assert_eq '{"image":3,"video":0,"audio":0}' "$MM_LIMIT_JSON" "explicit multimodal JSON"

reset_message_type_test_state
MESSAGE_TYPE=text+image
MM_LIMIT_JSON='{"image":1,"video":0,"audio":0}'
normalize_message_type_defaults
assert_eq 64 "$MM_IMAGE_LIMIT" "legacy image limit migration"
assert_eq '{"image":64,"video":0,"audio":0}' "$MM_LIMIT_JSON" "legacy multimodal limit migration"

reset_message_type_test_state
MESSAGE_TYPE=text+image
MM_IMAGE_LIMIT=0
if normalize_message_type_defaults 2>/dev/null; then
  echo "invalid image limit unexpectedly accepted" >&2
  exit 1
fi

MODE=normal
normalize_mode
assert_eq normal "$MODE" "explicit normal launch mode"

profile_without_mode=$(mktemp)
profile_with_mode=$(mktemp)
mtp_profile=$(mktemp)
model_dir=$(mktemp -d)
trap 'rm -f "$profile_without_mode" "$profile_with_mode" "$mtp_profile"; rm -rf "$model_dir"' EXIT
printf '%s\n' \
  'MODEL_FAMILY=qwen35moe' \
  'MODEL_VARIANT=fp8' \
  'KV_CACHE_DTYPE=float16' \
  'SPECULATIVE_METHOD=none' \
  'SPECULATIVE_TOKENS=0' > "$profile_without_mode"
printf '%s\n' \
  'MODE=fast' \
  'MODEL_FAMILY=qwen35moe' \
  'MODEL_VARIANT=fp8' \
  'KV_CACHE_DTYPE=float16' \
  'SPECULATIVE_METHOD=none' \
  'SPECULATIVE_TOKENS=0' > "$profile_with_mode"
printf '%s\n' \
  'MODEL_FAMILY=qwen35moe' \
  'MODEL_VARIANT=fp8' \
  'KV_CACHE_DTYPE=fp8' \
  'SPECULATIVE_METHOD=mtp' \
  'SPECULATIVE_TOKENS=4' > "$mtp_profile"

MODE=normal
apply_profile_overrides "$profile_without_mode"
assert_eq normal "$MODE" "mode survives mode-less profile"
apply_profile_overrides "$profile_with_mode"
assert_eq fast "$MODE" "profile may explicitly select fast"
apply_profile_overrides "$profile_without_mode"
assert_eq normal "$MODE" "mode-less profile restores launcher mode"

invalid_profile=$(mktemp)
printf '%s\n' 'MODEL_FAMILY=qwen35moe' 'ENABLE_PREFIX_CACHING=0' > "$invalid_profile"
if apply_profile_overrides "$invalid_profile" 2>/dev/null; then
  echo "runtime field unexpectedly accepted in route profile" >&2
  exit 1
fi
rm -f "$invalid_profile"

duplicate_profile=$(mktemp)
printf '%s\n' 'MODEL_FAMILY=qwen35moe' 'MODEL_FAMILY=qwen35' > "$duplicate_profile"
if apply_profile_overrides "$duplicate_profile" 2>/dev/null; then
  echo "duplicate profile key unexpectedly accepted" >&2
  exit 1
fi
rm -f "$duplicate_profile"

printf '%s\n' '{"model_type":"qwen3_5","max_position_embeddings":262144}' > "$model_dir/config.json"
MODEL_DIR=$model_dir
MAX_MODEL_LEN=524288
HF_OVERRIDES_JSON=
ENABLE_YARN=1
derive_yarn_overrides
grep -Fq '"rope_type":"yarn"' <<< "$HF_OVERRIDES_JSON"
grep -Fq '"factor":2.0' <<< "$HF_OVERRIDES_JSON"
grep -Fq '"original_max_position_embeddings":262144' <<< "$HF_OVERRIDES_JSON"
HF_OVERRIDES_JSON='{"text_config":{"max_position_embeddings":123456}}'
if derive_yarn_overrides 2>/dev/null; then
  echo "conflicting YaRN overrides unexpectedly accepted" >&2
  exit 1
fi

reset_spec_test_state() {
  unset SPECULATIVE_METHOD SPECULATIVE_TOKENS SPECULATIVE_MODEL MTP_K
  unset VLLM_KV_CACHE_LAYOUT SPECULATIVE_ATTENTION_BACKEND
  unset SPECULATIVE_KV_CACHE_DTYPE PER_REQUEST_SPEC_DECODE_METRICS
  unset LONG_PREFILL_TOKEN_THRESHOLD
  CONFIG_OVERRIDE_SOURCE=()
  CONFIG_OVERRIDE_UNSET=()
}

reset_spec_test_state
SPECULATIVE_METHOD=none
apply_speculative_runtime_defaults
assert_eq none "$(effective_speculative_method)" "none method"
assert_eq 0 "$(effective_speculative_tokens)" "none tokens"

reset_spec_test_state
SPECULATIVE_METHOD=mtp
SPECULATIVE_MODEL=/models/retained-dflash-draft
apply_speculative_runtime_defaults
assert_eq mtp "$(effective_speculative_method)" "mtp method"
assert_eq 3 "$(effective_speculative_tokens)" "mtp default tokens"
assert_eq 3 "$MTP_K" "mtp compatibility value"
[[ -z "${VLLM_KV_CACHE_LAYOUT:-}" ]]
validate_speculative_route

VLLM_KV_CACHE_LAYOUT=custom-layout
LONG_PREFILL_TOKEN_THRESHOLD=123
apply_speculative_runtime_defaults
assert_eq custom-layout "$VLLM_KV_CACHE_LAYOUT" "custom profile KV layout"
assert_eq 123 "$LONG_PREFILL_TOKEN_THRESHOLD" "custom profile prefill threshold"

reset_spec_test_state
SPECULATIVE_METHOD=none
SPECULATIVE_MODEL=/models/retained-dflash-draft
apply_speculative_runtime_defaults
validate_speculative_route
assert_eq 0 "$(effective_speculative_tokens)" "retained draft ignored for autoregressive"

apply_profile_overrides \
  "$mtp_profile"
assert_eq /models/retained-dflash-draft "$SPECULATIVE_MODEL" "draft survives profile switch"
assert_eq mtp "$(effective_speculative_method)" "profile switch selects MTP"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=1
TP_SIZE=2
VLLM_RUNTIME_BACKEND=native apply_speculative_runtime_defaults
assert_eq dflash "$(effective_speculative_method)" "dflash method"
assert_eq 7 "$(effective_speculative_tokens)" "dflash default tokens"
assert_eq BLHNC "$VLLM_KV_CACHE_LAYOUT" "dflash target KV layout"
assert_eq TRITON_ATTN "$SPECULATIVE_ATTENTION_BACKEND" "dflash attention backend"
assert_eq float16 "$SPECULATIVE_KV_CACHE_DTYPE" "dflash draft KV dtype"
assert_eq detailed "$PER_REQUEST_SPEC_DECODE_METRICS" "dflash metrics"

MODEL_DIR=/tmp/test-model
SERVED_NAME=test-model
PORT=18000
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_BATCHED_TOKENS=512
MODE=fast
MODEL_FAMILY=qwen
SPECULATIVE_MODEL=/models/dflash-draft
GENERATION_CONFIG=auto
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- '--per-request-spec-decode-metrics' <<<"$args"
grep -Fxq -- 'detailed' <<<"$args"
grep -Fxq -- '--generation-config' <<<"$args"
grep -Fxq -- 'auto' <<<"$args"

GENERATION_CONFIG=vllm
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- 'vllm' <<<"$args"
unset GENERATION_CONFIG
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
grep -Fxq -- 'auto' <<<"$args"

PER_REQUEST_SPEC_DECODE_METRICS=summary
apply_speculative_runtime_defaults
validate_spec_decode_metrics
assert_eq summary "$PER_REQUEST_SPEC_DECODE_METRICS" "metrics override"

SPECULATIVE_METHOD=none
SPECULATIVE_TOKENS=0
MTP_K=0
PER_REQUEST_SPEC_DECODE_METRICS=detailed
VLLM_ARGS=()
build_args 127.0.0.1
args=$(printf '%s\n' "${VLLM_ARGS[@]}")
! grep -Fxq -- '--per-request-spec-decode-metrics' <<<"$args"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=2
TP_SIZE=2
apply_speculative_runtime_defaults
assert_eq 2560 "$LONG_PREFILL_TOKEN_THRESHOLD" "dual-GPU DFlash C2 threshold"

reset_spec_test_state
SPECULATIVE_METHOD=dflash
MAX_NUM_SEQS=2
TP_SIZE=4
apply_speculative_runtime_defaults
assert_eq 512 "$LONG_PREFILL_TOKEN_THRESHOLD" "four-GPU DFlash C2 threshold"

echo "launcher_profile_schema_ok"
