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

required_keys=(
  MODEL_FAMILY
  MODEL_VARIANT
  QUANTIZATION
  KV_CACHE_DTYPE
  MAX_MODEL_LEN
  GPU_UTIL
  MAX_NUM_SEQS
  MESSAGE_TYPE
  SPECULATIVE_METHOD
  SPECULATIVE_TOKENS
)

allowed_key() {
  case "$1" in
    MODE|MODEL_FAMILY|MODEL_VARIANT|QUANTIZATION|KV_CACHE_DTYPE|MAX_MODEL_LEN|GPU_UTIL|\
MAX_BATCHED_TOKENS|MAX_NUM_SEQS|MESSAGE_TYPE|SPECULATIVE_METHOD|SPECULATIVE_TOKENS|ENABLE_YARN)
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

  gpu_util=$(read_profile_value "$file" GPU_UTIL)
  if ! awk -v value="$gpu_util" 'BEGIN { exit !(value ~ /^0([.][0-9]+)?$/ && value > 0 && value < 1) }'; then
    profile_error "$rel" "GPU_UTIL must be greater than 0 and lower than 1"
  fi

  message_type=$(read_profile_value "$file" MESSAGE_TYPE)
  case "$message_type" in
    text-only|text+image) ;;
    *) profile_error "$rel" "MESSAGE_TYPE must be text-only or text+image" ;;
  esac

  speculative_method=$(read_profile_value "$file" SPECULATIVE_METHOD)
  speculative_tokens=$(read_profile_value "$file" SPECULATIVE_TOKENS)
  profile_name=${rel##*/}
  if [[ "$profile_name" =~ mtp([0-9]+)- ]]; then
    mtp_label=${BASH_REMATCH[1]}
    if [[ "$speculative_tokens" != "$mtp_label" ]]; then
      profile_error "$rel" "filename MTP token count does not match SPECULATIVE_TOKENS=$speculative_tokens"
    fi
  fi
  if [[ "$profile_name" == nomtp-* && "$speculative_method" != none ]]; then
    profile_error "$rel" "nomtp filename requires SPECULATIVE_METHOD=none"
  fi
  if [[ "$profile_name" == dflash* && "$speculative_method" != dflash ]]; then
    profile_error "$rel" "dflash filename requires SPECULATIVE_METHOD=dflash"
  fi
  if [[ "$profile_name" =~ -([1-9][0-9]*)x[0-9]+[kK]- ]]; then
    expected_seqs=${BASH_REMATCH[1]}
    [[ "$expected_seqs" == "$(read_profile_value "$file" MAX_NUM_SEQS)" ]] || \
      profile_error "$rel" "filename concurrency does not match MAX_NUM_SEQS"
  fi
  if [[ "$profile_name" =~ -[1-9][0-9]*x([1-9][0-9]*)[kK]- ]]; then
    actual_context=$(read_profile_value "$file" MAX_MODEL_LEN)
    # Context labels are decimal thousands; uppercase K is canonical.
    expected_context=$((10#${BASH_REMATCH[1]}))
    actual_context_k=$((actual_context / 1000))
    legacy_context=$((expected_context * 1024))
    legacy_delta=$((actual_context - legacy_context))
    (( legacy_delta < 0 )) && legacy_delta=$((-legacy_delta))
    if [[ "$PROFILE_DIR" == "$ROOT/profiles" ]]; then
      context_ok=$(( actual_context_k == expected_context || legacy_delta <= 1024 ))
    else
      context_ok=$(( actual_context_k == expected_context ))
    fi
    (( context_ok )) || \
      profile_error "$rel" "filename context does not match MAX_MODEL_LEN"
  fi
  case "$profile_name" in
    *fp16kv-*) [[ "$kv" == float16 ]] || profile_error "$rel" "fp16kv filename requires KV_CACHE_DTYPE=float16" ;;
    *fp8kv-*) [[ "$kv" == fp8 ]] || profile_error "$rel" "fp8kv filename requires KV_CACHE_DTYPE=fp8" ;;
    *tq4nc-*) [[ "$kv" == turboquant_4bit_nc ]] || profile_error "$rel" "tq4nc filename requires KV_CACHE_DTYPE=turboquant_4bit_nc" ;;
    *tqk8v4-*) [[ "$kv" == turboquant_k8v4 ]] || profile_error "$rel" "tqk8v4 filename requires KV_CACHE_DTYPE=turboquant_k8v4" ;;
  esac
  case "$profile_name" in
    *text-only.env) [[ "$message_type" == text-only ]] || profile_error "$rel" "text-only filename requires MESSAGE_TYPE=text-only" ;;
    *text-image.env) [[ "$message_type" == text+image ]] || profile_error "$rel" "text-image filename requires MESSAGE_TYPE=text+image" ;;
  esac
  case "$speculative_method" in
    none)
      [[ "$speculative_tokens" == 0 ]] || \
        profile_error "$rel" "SPECULATIVE_METHOD=none requires SPECULATIVE_TOKENS=0"
      ;;
    mtp|dflash)
      [[ "$speculative_tokens" =~ ^[1-9][0-9]*$ ]] || \
        profile_error "$rel" "$speculative_method requires positive SPECULATIVE_TOKENS"
      ;;
    *)
      profile_error "$rel" "SPECULATIVE_METHOD must be none, mtp, or dflash"
      ;;
  esac

  yarn=$(read_profile_value "$file" ENABLE_YARN)
  if [[ -n "$yarn" && "$yarn" != 0 && "$yarn" != 1 ]]; then
    profile_error "$rel" "ENABLE_YARN must be 0 or 1 when present"
  fi
  case "${rel##*/}" in
    yarn-*)
      [[ "$yarn" == 1 ]] || profile_error "$rel" "YaRN profile filename requires ENABLE_YARN=1"
      ;;
    *)
      [[ -z "$yarn" || "$yarn" == 0 ]] || profile_error "$rel" "ENABLE_YARN=1 requires a yarn-* filename"
      ;;
  esac
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
