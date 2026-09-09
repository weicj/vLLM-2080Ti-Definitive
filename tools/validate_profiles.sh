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

profile_key_is_allowed() {
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
VLLM_ALLOW_LONG_MAX_MODEL_LEN|VLLM_FORCE_NVFP4_W4A16|VLLM_PLE_CPU_OFFLOAD|\
VLLM_STATIC_PP_SINGLE_TOKEN|\
VLLM_INT8KV_FA_PREFILL|\
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
  tp=$(read_profile_value "$file" TP_SIZE)
  pp=$(read_profile_value "$file" PP_SIZE)
  pp_partition=$(read_profile_value "$file" VLLM_PP_LAYER_PARTITION)
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

  if [[ -n "$tp" || -n "$pp" || -n "$pp_partition" ]]; then
    if [[ -n "$tp" && ! "$tp" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR $rel: TP_SIZE must be a positive integer" >&2
      ((errors += 1))
    fi
    if [[ -n "$pp" && ! "$pp" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR $rel: PP_SIZE must be a positive integer" >&2
      ((errors += 1))
    fi

    # launcher.sh defaults PP_SIZE to 1 and derives a missing TP_SIZE, so a
    # route may legitimately provide only one of the two sizes.
    effective_pp=${pp:-1}
    if [[ -n "$pp_partition" && "$effective_pp" =~ ^[1-9][0-9]*$ ]]; then
      IFS=',' read -r -a partition_parts <<< "$pp_partition"
      if (( ${#partition_parts[@]} != effective_pp )); then
        echo "ERROR $rel: VLLM_PP_LAYER_PARTITION must contain PP_SIZE=$effective_pp entries" >&2
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

if ((errors > 0)); then
  echo "profile_validation_failed total=$total errors=$errors" >&2
  exit 1
fi

echo "profile_validation_ok total=$total"
