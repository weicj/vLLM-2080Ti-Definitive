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
    MODEL_DIR|PROFILE_DIR|PROFILE|PORT|SERVICE_SCOPE|GPU_DEVICES|\
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

allowed_key() {
  case "$1" in
    SERVED_NAME|COMPATIBLE_MODES|MODEL_FAMILY|PROFILE_GROUP|MODEL_VARIANT|\
TP_SIZE|PP_SIZE|VLLM_PP_LAYER_PARTITION|\
QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN|KV_CACHE_MEMORY_BYTES|GPU_UTIL|\
MAX_BATCHED_TOKENS|LONG_PREFILL_TOKEN_THRESHOLD|\
MAX_NUM_SEQS|PREFILL_BATCH_BARRIER|DISABLE_PREFIX_CACHING|MTP_K|\
MESSAGE_TYPE|MM_LIMIT_JSON|LANGUAGE_MODEL_ONLY|\
SKIP_MM_PROFILING|HF_OVERRIDES_JSON|ADDITIONAL_CONFIG_JSON|\
SPECULATIVE_CONFIG|ATTENTION_BACKEND|DISABLE_HYBRID_KV_CACHE_MANAGER|\
DISABLE_CUSTOM_ALL_REDUCE|\
VLLM_ALLOW_LONG_MAX_MODEL_LEN|VLLM_INT8KV_FA_PREFILL|\
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

expected_profiles=(
  2xT10/qwen27b/w4a16/mtp3-fp8kv-1x262K-text-only.env
  2xT10/qwen27b/w4a16/mtp3-tq4nc-2x220K-text-only.env
  2xT10/qwen27b/w4a16/mtp3-tqk8v4-1x196K-text-image.env
  2xT10/qwen27b/w4a16/mtp3-tqk8v4-2x155K-text-only.env
  2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x121K-text-image.env
  2x2080Ti/qwen27b/w8a16/nomtp-fp16kv-1x176K-text-only.env
  2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env
  2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x186K-text-image.env
  2x2080Ti/qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env
  2x2080Ti/qwen27b/w8a16/yarn-fp8kv-1x338K-text-only.env
  2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env
  2x2080Ti/qwen27b/w4a16/yarn-fp8kv-1x524K-text-only.env
  2x2080Ti/qwen27b/w4a16/mtp4-fp8kv-2x229K-text-only.env
  2x2080Ti/qwen27b/w4a16/dflash-fp8kv-2x176K-text-only.env
  2x2080Ti/qwen27b/w4a16/mtp4-tq4nc-3x262K-text-only.env
  2x2080Ti/qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env
  2x2080Ti/qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env
  4xT10/qwen27b/w4a16/mtp4-fp16kv-1x262K-text-image.env
  4xT10/qwen27b/w4a16/dflash2-fp16kv-1x262K-text-only.env
  4xT10/qwen27b/w4a16/dflash2-fp8kv-2x262K-text-only.env
  4xT10/qwen27b/w4a16/dflash2-tqk8v4-4x220K-text-only.env
  4xT10/qwen27b/w4a16/mtp4-fp8kv-2x262K-text-image.env
  4xT10/qwen27b/w8a16/dflash2-fp16kv-1x240K-text-image.env
  4xT10/qwen27b/w8a16/dflash2-fp16kv-1x262K-text-only.env
  4xT10/qwen27b/w8a16/dflash2-fp8kv-2x220K-text-only.env
  4xT10/qwen27b/w8a16/mtp4-fp8kv-2x262K-text-image.env
  4xT10/qwen27b/w8a16/mtp4-fp16kv-1x262K-text-image.env
)
mapfile -t expected_profiles < <(printf '%s\n' "${expected_profiles[@]}" | sort)

profile_error() {
  local rel=$1
  shift
  echo "ERROR $rel: $*" >&2
  ((errors += 1))
}

while IFS= read -r -d '' file; do
  ((total += 1))
  rel=${file#"$PROFILE_DIR"/}
  mode=$(read_profile_value "$file" MODE)
  compatible_modes=$(read_profile_value "$file" COMPATIBLE_MODES)
  kv=$(read_profile_value "$file" KV_CACHE_DTYPE)
  mtp=$(read_profile_value "$file" MTP_K)
  tp=$(read_profile_value "$file" TP_SIZE)
  pp=$(read_profile_value "$file" PP_SIZE)
  pp_partition=$(read_profile_value "$file" VLLM_PP_LAYER_PARTITION)
  has_safe=0

  if [[ "/$rel/" == */fast/* || "/$rel/" == */normal/* ]]; then
    profile_error "$rel" "fast/normal must be selected by the launcher, not profile directories"
  fi

  invalid_line=$(sed -nE '/^[[:space:]]*($|#)/d; /^[A-Za-z_][A-Za-z0-9_]*=.*/d; =' "$file" | head -n 1)
  if [[ -n "$invalid_line" ]]; then
    profile_error "$rel" "invalid profile syntax: $invalid_line"
  fi

  duplicate_keys=$(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort | uniq -d)
  if [[ -n "$duplicate_keys" ]]; then
    while IFS= read -r key; do
      [[ -n "$key" ]] && profile_error "$rel" "duplicate key $key"
    done <<< "$duplicate_keys"
  fi

  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    if ! allowed_key "$key"; then
      profile_error "$rel" "$key is not part of the route-profile schema"
    fi
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)

  for key in "${required_keys[@]}"; do
    value=$(read_profile_value "$file" "$key")
    if [[ -z "$value" ]]; then
      profile_error "$rel" "$key is required and must not be empty"
    fi
  done

  mode=$(read_profile_value "$file" MODE)
  if [[ -n "$mode" && "$mode" != fast && "$mode" != normal ]]; then
    profile_error "$rel" "MODE may be omitted or set to fast/normal, got $mode"
  fi

  kv=$(read_profile_value "$file" KV_CACHE_DTYPE)
  case "$kv" in
    float16|fp8|turboquant_4bit_nc|turboquant_k8v4) ;;
    auto|default|"")
      profile_error "$rel" "KV_CACHE_DTYPE must explicitly name the stored KV precision"
      ;;
    *)
      profile_error "$rel" "unsupported KV_CACHE_DTYPE=$kv"
      ;;
  esac

  for key in MAX_MODEL_LEN MAX_NUM_SEQS; do
    value=$(read_profile_value "$file" "$key")
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      profile_error "$rel" "$key must be a positive integer"
    fi
  done
  max_batched_tokens=$(read_profile_value "$file" MAX_BATCHED_TOKENS)
  if [[ -n "$max_batched_tokens" && ! "$max_batched_tokens" =~ ^[1-9][0-9]*$ ]]; then
    profile_error "$rel" "MAX_BATCHED_TOKENS must be a positive integer when present"
  fi

  if [[ -n "$tp" && ! "$tp" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR $rel: TP_SIZE must be a positive integer" >&2
    ((errors += 1))
  fi
  if [[ -n "$pp" && ! "$pp" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR $rel: PP_SIZE must be a positive integer" >&2
    ((errors += 1))
  fi
  if [[ -n "$pp_partition" ]]; then
    if [[ "$pp_partition" == ,* || "$pp_partition" == *, || "$pp_partition" == *,,* ]]; then
      echo "ERROR $rel: VLLM_PP_LAYER_PARTITION must not contain empty entries" >&2
      ((errors += 1))
    elif [[ "${pp:-1}" =~ ^[1-9][0-9]*$ ]]; then
      IFS=',' read -r -a partition_parts <<< "$pp_partition"
      if (( ${#partition_parts[@]} != ${pp:-1} )); then
        echo "ERROR $rel: VLLM_PP_LAYER_PARTITION must contain PP_SIZE=${pp:-1} entries" >&2
        ((errors += 1))
      else
        for partition in "${partition_parts[@]}"; do
          if [[ ! "$partition" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR $rel: every VLLM_PP_LAYER_PARTITION entry must be a positive integer, got '$partition'" >&2
            ((errors += 1))
          fi
        done
      fi
    fi
  fi
done < <(find "$PROFILE_DIR" -type f -name '*.env' -print0 | sort -z)

if [[ "$PROFILE_DIR" == "$ROOT/profiles" ]]; then
  mapfile -t actual_profiles < <(find "$PROFILE_DIR" -type f -name '*.env' -printf '%P\n' | sort)
  if (( ${#actual_profiles[@]} != ${#expected_profiles[@]} )); then
    profile_error "<profile-matrix>" "expected exactly ${#expected_profiles[@]} official profiles, found ${#actual_profiles[@]}"
  else
    for index in "${!expected_profiles[@]}"; do
      if [[ "${actual_profiles[$index]}" != "${expected_profiles[$index]}" ]]; then
        profile_error "<profile-matrix>" "unexpected official profile at index $index: ${actual_profiles[$index]}"
      fi
    done
  fi
fi

if ((errors > 0)); then
  echo "profile_validation_failed total=$total errors=$errors" >&2
  exit 1
fi

echo "profile_validation_ok total=$total"
