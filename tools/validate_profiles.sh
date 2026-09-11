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

json_config_field() {
  local payload=$1
  local field=$2
  python3 - "$field" "$payload" <<'PY'
import json
import sys

field = sys.argv[1]
payload = sys.argv[2]

try:
    data = json.loads(payload)
except json.JSONDecodeError:
    raise SystemExit(2)

value = data.get(field)
if value is None:
    raise SystemExit(0)
print(value)
PY
}

normalize_speculative_method_value() {
  case "${1,,}" in
    ""|none|off|disabled) ;;
    mtp) echo mtp ;;
    dflash) echo dflash ;;
    *)
      printf '%s\n' "${1,,}"
      ;;
  esac
}

infer_speculative_method_from_model_ref() {
  local ref=${1:-}
  ref=${ref,,}
  case "$ref" in
    *dflash*)
      printf 'dflash\n'
      ;;
  esac
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
    SERVED_NAME|COMPATIBLE_MODES|MODEL_FAMILY|PROFILE_GROUP|MODEL_VARIANT|PLE_PLACEMENT|\
TP_SIZE|PP_SIZE|VLLM_PP_LAYER_PARTITION|VLLM_FORCE_NVFP4_W4A16|VLLM_PLE_CPU_OFFLOAD|\
QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN|GPU_UTIL|MAX_BATCHED_TOKENS|\
MAX_NUM_SEQS|NO_ASYNC_SCHEDULING|MTP_K|SPECULATIVE_METHOD|SPECULATIVE_MODEL|SPECULATIVE_TOKENS|\
SPECULATIVE_DRAFT_TP_SIZE|SPECULATIVE_MAX_MODEL_LEN|\
SPECULATIVE_ATTENTION_BACKEND|SPECULATIVE_DISABLE_PADDED_DRAFTER_BATCH|\
MESSAGE_TYPE|MM_LIMIT_JSON|LANGUAGE_MODEL_ONLY|\
SKIP_MM_PROFILING|HF_OVERRIDES_JSON|ADDITIONAL_CONFIG_JSON|\
SPECULATIVE_CONFIG|ATTENTION_BACKEND|DISABLE_HYBRID_KV_CACHE_MANAGER|\
CUSTOM_ALL_REDUCE_MODE|DISABLE_CUSTOM_ALL_REDUCE|\
VLLM_ALLOW_LONG_MAX_MODEL_LEN|VLLM_TURBOQUANT_SPEC_DECODE_CHUNK_SIZE|VLLM_INT8KV_FA_PREFILL|\
VLLM_INT8KV_FA_CONTINUATION_DEQUANT|VLLM_INT8KV_FA_CASCADE_DEQUANT|\
    VLLM_INT8KV_FA_CASCADE_TILE_TOKENS|DISABLE_PREFIX_CACHING)
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
  raw_spec_method=$(read_profile_value "$file" SPECULATIVE_METHOD)
  raw_spec_model=$(read_profile_value "$file" SPECULATIVE_MODEL)
  raw_spec_tokens=$(read_profile_value "$file" SPECULATIVE_TOKENS)
  raw_spec_draft_tp=$(read_profile_value "$file" SPECULATIVE_DRAFT_TP_SIZE)
  raw_spec_max_model_len=$(read_profile_value "$file" SPECULATIVE_MAX_MODEL_LEN)
  raw_spec_backend=$(read_profile_value "$file" SPECULATIVE_ATTENTION_BACKEND)
  raw_spec_disable_padded=$(read_profile_value "$file" SPECULATIVE_DISABLE_PADDED_DRAFTER_BATCH)
  spec_method=$(normalize_speculative_method_value "$raw_spec_method")
  spec_model=$raw_spec_model
  spec_tokens=$raw_spec_tokens
  spec_json=$(read_profile_value "$file" SPECULATIVE_CONFIG)
  custom_all_reduce_mode=$(read_profile_value "$file" CUSTOM_ALL_REDUCE_MODE)
  disable_custom_all_reduce=$(read_profile_value "$file" DISABLE_CUSTOM_ALL_REDUCE)
  model_family=$(read_profile_value "$file" MODEL_FAMILY)
  ple_placement=$(read_profile_value "$file" PLE_PLACEMENT)
  legacy_ple_offload=$(read_profile_value "$file" VLLM_PLE_CPU_OFFLOAD)
  has_safe=0

  if [[ -n "$ple_placement" ]]; then
    case "${ple_placement,,}" in
      disk|cpu|gpu)
        ;;
      *)
        echo "ERROR $rel: PLE_PLACEMENT must be disk, cpu, or gpu" >&2
        ((errors += 1))
        ;;
    esac
    if [[ "$model_family" != qwen4* ]]; then
      echo "ERROR $rel: PLE_PLACEMENT is only valid for MODEL_FAMILY=qwen4" >&2
      ((errors += 1))
    fi
    if [[ -n "$legacy_ple_offload" ]]; then
      echo "ERROR $rel: PLE_PLACEMENT cannot be combined with VLLM_PLE_CPU_OFFLOAD" >&2
      ((errors += 1))
    fi
  fi

  if [[ -n "$custom_all_reduce_mode" ]]; then
    case "${custom_all_reduce_mode,,}" in
      auto|off)
        ;;
      *)
        echo "ERROR $rel: CUSTOM_ALL_REDUCE_MODE must be auto or off" >&2
        ((errors += 1))
        ;;
    esac
    if [[ -n "$disable_custom_all_reduce" ]]; then
      echo "ERROR $rel: CUSTOM_ALL_REDUCE_MODE cannot be combined with DISABLE_CUSTOM_ALL_REDUCE" >&2
      ((errors += 1))
    fi
  fi

  if [[ -n "$mtp" && ! "$mtp" =~ ^[0-9]+$ ]]; then
    echo "ERROR $rel: MTP_K must be a non-negative integer, got $mtp" >&2
    ((errors += 1))
  fi

  if [[ -n "$tp" || -n "$pp" || -n "$pp_partition" ]]; then
    if [[ ! "$tp" =~ ^[1-9][0-9]*$ || ! "$pp" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR $rel: TP_SIZE and PP_SIZE must both be positive integers when a PP layout is specified" >&2
      ((errors += 1))
    elif [[ -n "$pp_partition" ]]; then
      IFS=',' read -r -a partition_parts <<< "$pp_partition"
      if (( ${#partition_parts[@]} != pp )); then
        echo "ERROR $rel: VLLM_PP_LAYER_PARTITION must contain PP_SIZE=$pp entries" >&2
        ((errors += 1))
      fi
    fi
  fi

  if [[ -n "$spec_json" ]]; then
    json_config_field "$spec_json" method >/dev/null 2>&1 || spec_json_status=$?
    if [[ "${spec_json_status:-0}" == "2" ]]; then
      echo "ERROR $rel: SPECULATIVE_CONFIG is not valid JSON" >&2
      ((errors += 1))
      spec_json=
    fi
    unset spec_json_status
  fi

  if [[ -n "$spec_json" && ( -n "$raw_spec_method" || -n "$raw_spec_model" || -n "$raw_spec_tokens" || -n "$raw_spec_draft_tp" || -n "$raw_spec_max_model_len" || -n "$raw_spec_backend" || -n "$raw_spec_disable_padded" || -n "$mtp" ) ]]; then
    echo "ERROR $rel: SPECULATIVE_CONFIG must not be mixed with shortcut speculative fields or MTP_K" >&2
    ((errors += 1))
  fi

  if [[ -n "$spec_json" ]]; then
    if [[ -z "$spec_method" ]]; then
      spec_method=$(normalize_speculative_method_value "$(json_config_field "$spec_json" method 2>/dev/null || true)")
    fi
    if [[ -z "$spec_model" ]]; then
      spec_model=$(json_config_field "$spec_json" model 2>/dev/null || true)
    fi
    if [[ -z "$spec_tokens" ]]; then
      spec_tokens=$(json_config_field "$spec_json" num_speculative_tokens 2>/dev/null || true)
    fi
  fi

  if [[ -z "$spec_method" && -n "$spec_model" ]]; then
    spec_method=$(infer_speculative_method_from_model_ref "$spec_model")
  fi
  if [[ -z "$spec_method" && "$mtp" =~ ^[0-9]+$ ]] && (( mtp > 0 )); then
    spec_method=mtp
  fi
  if [[ -z "$spec_tokens" ]]; then
    spec_tokens=$mtp
  fi

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

  if [[ -z "$spec_method" ]]; then
    if [[ -n "$raw_spec_model" || -n "$raw_spec_tokens" || -n "$raw_spec_draft_tp" || -n "$raw_spec_max_model_len" || -n "$raw_spec_backend" || -n "$raw_spec_disable_padded" ]]; then
      echo "ERROR $rel: speculative shortcut fields require SPECULATIVE_METHOD=mtp|dflash or SPECULATIVE_CONFIG" >&2
      ((errors += 1))
    fi
  else
    case "$spec_method" in
      mtp|dflash)
        ;;
      *)
        echo "ERROR $rel: SPECULATIVE_METHOD=$spec_method is not supported by launcher shortcuts" >&2
        ((errors += 1))
        ;;
    esac
  fi

  if [[ -n "$spec_method" ]] && [[ ! "$spec_tokens" =~ ^[0-9]+$ || "$spec_tokens" == "0" ]]; then
    echo "ERROR $rel: speculative decode requires a positive speculative token count" >&2
    ((errors += 1))
  fi

  if [[ "$spec_method" == "mtp" ]]; then
    if [[ -n "$raw_spec_model" || -n "$raw_spec_draft_tp" || -n "$raw_spec_max_model_len" || -n "$raw_spec_backend" || -n "$raw_spec_disable_padded" ]]; then
      echo "ERROR $rel: MTP shortcut must not set DFlash-only speculative fields" >&2
      ((errors += 1))
    fi
  fi

  if [[ "$spec_method" == "dflash" && -n "$mtp" && "$mtp" != "0" ]]; then
    echo "ERROR $rel: DFlash profiles must not set MTP_K alongside SPECULATIVE_METHOD=dflash" >&2
    ((errors += 1))
  fi

  if [[ -n "$raw_spec_draft_tp" ]] && ([[ ! "$raw_spec_draft_tp" =~ ^[0-9]+$ ]] || (( raw_spec_draft_tp <= 0 ))); then
    echo "ERROR $rel: SPECULATIVE_DRAFT_TP_SIZE must be a positive integer" >&2
    ((errors += 1))
  fi

  if [[ -n "$raw_spec_max_model_len" ]] && ([[ ! "$raw_spec_max_model_len" =~ ^[0-9]+$ ]] || (( raw_spec_max_model_len <= 0 ))); then
    echo "ERROR $rel: SPECULATIVE_MAX_MODEL_LEN must be a positive integer" >&2
    ((errors += 1))
  fi

  tq_spec_chunk=$(read_profile_value "$file" VLLM_TURBOQUANT_SPEC_DECODE_CHUNK_SIZE)
  if [[ -n "$tq_spec_chunk" ]] && ([[ ! "$tq_spec_chunk" =~ ^[0-9]+$ ]] || (( tq_spec_chunk != 1 && tq_spec_chunk != 2 && tq_spec_chunk != 4 && tq_spec_chunk != 8 ))); then
    echo "ERROR $rel: VLLM_TURBOQUANT_SPEC_DECODE_CHUNK_SIZE must be one of 1, 2, 4, or 8" >&2
    ((errors += 1))
  fi

  if [[ "$spec_method" == "dflash" && -z "$spec_model" ]]; then
    echo "ERROR $rel: DFlash profiles require SPECULATIVE_MODEL or a JSON model entry" >&2
    ((errors += 1))
  fi

  if (( has_safe )); then
    case "$kv" in
      ""|fp16|default|auto)
        ;;
      *)
        if [[ "$spec_tokens" =~ ^[0-9]+$ ]] && (( spec_tokens > 0 )); then
          echo "ERROR $rel: safe quantized-KV profiles must not enable speculative decode, got spec_tokens=$spec_tokens KV=$kv" >&2
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
