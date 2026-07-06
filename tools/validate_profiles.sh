#!/usr/bin/env bash
set -euo pipefail

ROOT=${STABLE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PROFILE_DIR=${PROFILE_DIR:-"$ROOT/profiles"}

if [[ ! -d "$PROFILE_DIR" ]]; then
  echo "profile directory not found: $PROFILE_DIR" >&2
  exit 2
fi

read_profile_value() {
  local file=$1
  local key=$2
  awk -F= -v key="$key" '
    $1 == key {
      value = substr($0, index($0, "=") + 1)
      gsub(/^[ \t]+|[ \t]+$/, "", value)
      gsub(/^'\''|'\''$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "$file"
}

profile_key_is_global() {
  case "$1" in
    MODEL_DIR|PROFILE_DIR|PROFILE|PORT|SERVICE_SCOPE|GPU_DEVICES|TP_SIZE|\
CHAT_TEMPLATE_FILE|CHAT_TEMPLATE_PRESET|TEMPLATE_DIR|REASONING_PARSER|\
DEFAULT_CHAT_TEMPLATE_KWARGS|REASONING_MODE|REASONING_BUDGET|\
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

profile_key_is_allowed() {
  case "$1" in
    SERVED_NAME|COMPATIBLE_MODES|MODEL_FAMILY|PROFILE_GROUP|MODEL_VARIANT|\
QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN|KV_CACHE_MEMORY_BYTES|GPU_UTIL|\
MAX_BATCHED_TOKENS|\
KV_CACHE_DTYPE_SKIP_LAYERS|\
MAX_NUM_SEQS|MTP_K|MESSAGE_TYPE|MM_LIMIT_JSON|LANGUAGE_MODEL_ONLY|\
SKIP_MM_PROFILING|HF_OVERRIDES_JSON|ADDITIONAL_CONFIG_JSON|\
SPECULATIVE_CONFIG|ATTENTION_BACKEND|DISABLE_HYBRID_KV_CACHE_MANAGER|\
DISABLE_CUSTOM_ALL_REDUCE|\
VLLM_ALLOW_LONG_MAX_MODEL_LEN|VLLM_INT8KV_FA_PREFILL|\
VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT|VLLM_INT8KV_FA_DIRECT_PAGED|\
VLLM_INT8KV_ALIGNED_HEAD_STRIDE|\
VLLM_INT8KV_FA_CONTINUATION_DEQUANT|VLLM_INT8KV_FA_CASCADE_DEQUANT|\
VLLM_INT8KV_FA_CASCADE_TILE_TOKENS|\
VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE|\
VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS|\
VLLM_TURBOQUANT_MAX_KV_SPLITS|VLLM_TURBOQUANT_DECODE_BLOCK_KV)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

total=0
errors=0

while IFS= read -r -d '' file; do
  ((total += 1))
  rel=${file#"$PROFILE_DIR"/}
  mode=$(read_profile_value "$file" MODE)
  compatible_modes=$(read_profile_value "$file" COMPATIBLE_MODES)
  kv=$(read_profile_value "$file" KV_CACHE_DTYPE)
  mtp=$(read_profile_value "$file" MTP_K)
  has_safe=0

  if [[ -n "$mode" ]]; then
    echo "ERROR $rel: MODE should not be pinned inside a profile; use COMPATIBLE_MODES or launcher MODE" >&2
    ((errors += 1))
  fi

  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    if profile_key_is_global "$key"; then
      echo "ERROR $rel: $key is a global launcher setting and must not be stored in a route profile" >&2
      ((errors += 1))
    elif ! profile_key_is_allowed "$key"; then
      echo "ERROR $rel: $key is not an allowed route profile setting" >&2
      ((errors += 1))
    fi
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)

  if [[ -z "$compatible_modes" ]]; then
    echo "ERROR $rel: COMPATIBLE_MODES is required" >&2
    ((errors += 1))
  else
    IFS=',' read -r -a mode_parts <<< "$compatible_modes"
    for compatible_mode in "${mode_parts[@]}"; do
      compatible_mode=${compatible_mode//[[:space:]]/}
      case "$compatible_mode" in
        safe|normal|fast|aggressive)
          [[ "$compatible_mode" == "safe" ]] && has_safe=1
        ;;
      *)
        echo "ERROR $rel: COMPATIBLE_MODES must contain only safe/normal/fast/aggressive, got $compatible_modes" >&2
        ((errors += 1))
        ;;
    esac
    done
  fi

  if (( has_safe )); then
    case "$kv" in
      ""|fp16|default|auto)
        ;;
      *)
        if [[ "$mtp" =~ ^[0-9]+$ ]] && (( mtp > 0 )); then
          echo "ERROR $rel: safe quantized-KV profiles must set MTP_K=0, got MTP_K=$mtp KV=$kv" >&2
          ((errors += 1))
        fi
        ;;
    esac
  fi
done < <(find "$PROFILE_DIR" -type f -name '*.env' -print0 | sort -z)

if ((errors > 0)); then
  echo "profile_validation_failed total=$total errors=$errors" >&2
  exit 1
fi

echo "profile_validation_ok total=$total"
