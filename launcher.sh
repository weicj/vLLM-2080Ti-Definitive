#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "$(basename "$SCRIPT_DIR")" == "launcher" && -d "$SCRIPT_DIR/../profiles" ]]; then
  MANAGER_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
else
  MANAGER_ROOT="$SCRIPT_DIR"
fi
PROJECT_ROOT="$MANAGER_ROOT"
PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$MANAGER_ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"
RUNTIME_ROOT=${RUNTIME_ROOT:-"$MANAGER_ROOT"}
PROFILE_DIR=${PROFILE_DIR:-"$MANAGER_ROOT/profiles"}
TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
LOG_DIR=${LOG_DIR:-"$MANAGER_ROOT/run-logs"}
STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
STAMP=$(date +%Y%m%d-%H%M%S)
VERSION=${VERSION:-$FORK_RELEASE}
MENU_DIGIT_TIMEOUT=${LAUNCHER_MENU_DIGIT_TIMEOUT:-0.8}

banner() {
  cat <<EOF
============================================================
 $PROJECT_NAME v$VERSION
 Service manager
 Runtime: $RUNTIME_IDENTITY
 Author: $PROJECT_AUTHOR
============================================================
EOF
}

pid_is_running() {
  local pid=${1:-}
  [[ -n "$pid" && -d "/proc/$pid" ]]
}

pid_file_service_name() {
  local pid_file=$1
  local name
  name=$(basename "$pid_file" .pid)
  name=${name#vllm-}
  printf '%s\n' "$name"
}

current_service_info() {
  local pid_file pid name

  if [[ -n "${LAST_PID_FILE:-}" && -f "${LAST_PID_FILE:-}" ]]; then
    pid=$(cat "$LAST_PID_FILE" 2>/dev/null || true)
    if pid_is_running "$pid"; then
      name=$(pid_file_service_name "$LAST_PID_FILE")
      printf '%s\t%s\t%s\n' "$LAST_PID_FILE" "$pid" "$name"
      return 0
    fi
  fi

  [[ -d "$LOG_DIR" ]] || return 1
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if pid_is_running "$pid"; then
      name=$(pid_file_service_name "$pid_file")
      printf '%s\t%s\t%s\n' "$pid_file" "$pid" "$name"
      return 0
    fi
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  return 1
}

pid_arg_value() {
  local pid=$1
  local flag=$2
  local part want_next=0

  [[ -r "/proc/$pid/cmdline" ]] || return 1
  while IFS= read -r -d '' part; do
    if (( want_next )); then
      printf '%s\n' "$part"
      return 0
    fi
    if [[ "$part" == "$flag" ]]; then
      want_next=1
    fi
  done <"/proc/$pid/cmdline"
  return 1
}

pid_has_arg() {
  local pid=$1
  local flag=$2
  local part

  [[ -r "/proc/$pid/cmdline" ]] || return 1
  while IFS= read -r -d '' part; do
    [[ "$part" == "$flag" ]] && return 0
  done <"/proc/$pid/cmdline"
  return 1
}

pid_prefix_cache_label() {
  local pid=$1
  if pid_has_arg "$pid" --no-enable-prefix-caching; then
    printf 'disabled\n'
  elif pid_has_arg "$pid" --enable-prefix-caching; then
    printf 'enabled\n'
  else
    printf 'auto\n'
  fi
}

pid_prompt_details_label() {
  local pid=$1
  if pid_has_arg "$pid" --enable-prompt-tokens-details; then
    printf 'enabled\n'
  else
    printf 'disabled\n'
  fi
}

service_api_root() {
  local pid_file=$1
  local pid=$2
  local api port

  if [[ -n "${LAST_API_LOCAL:-}" && "$pid_file" == "${LAST_PID_FILE:-}" ]]; then
    api=${LAST_API_LOCAL%/}
    printf '%s\n' "${api%/v1}"
    return 0
  fi

  port=$(pid_arg_value "$pid" --port 2>/dev/null || true)
  port=${port:-${PORT:-8000}}
  printf 'http://127.0.0.1:%s\n' "$port"
}

service_log_file() {
  local pid_file=$1
  local name=$2
  local log_file

  if [[ -n "${LAST_LOG_FILE:-}" && -f "${LAST_LOG_FILE:-}" && "$pid_file" == "${LAST_PID_FILE:-}" ]]; then
    printf '%s\n' "$LAST_LOG_FILE"
    return 0
  fi

  log_file=$(
    find "$LOG_DIR" -maxdepth 1 -type f -name "vllm-${name}-*.log" -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      awk 'NR == 1 { $1=""; sub(/^ /, ""); print }'
  )
  [[ -n "$log_file" ]] || return 1
  printf '%s\n' "$log_file"
}

kv_cache_total_tokens_from_log() {
  local log_file=$1
  local total

  [[ -f "$log_file" ]] || return 1
  total=$(sed -n 's/.*GPU KV cache size:[[:space:]]*\([0-9,]\+\) tokens.*/\1/p' "$log_file" | tail -n 1)
  total=${total//,/}
  [[ "$total" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$total"
}

kv_cache_usage_ratio_from_metrics() {
  local api_root=$1
  local metrics ratio

  metrics=$(curl -fsS --max-time 2 "${api_root%/}/metrics" 2>/dev/null || true)
  [[ -n "$metrics" ]] || return 1
  ratio=$(
    awk '
      /^#/ { next }
      $1 ~ /^vllm:kv_cache_usage_perc(\{|$)/ { print $NF; exit }
      $1 ~ /^vllm:gpu_cache_usage_perc(\{|$)/ { print $NF; exit }
    ' <<<"$metrics"
  )
  [[ "$ratio" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  printf '%s\n' "$ratio"
}

render_kv_cache_status() {
  local pid_file=$1
  local pid=$2
  local name=$3
  local label_width=${4:-8}
  local api_root log_file total ratio used pct

  api_root=$(service_api_root "$pid_file" "$pid")
  log_file=$(service_log_file "$pid_file" "$name" 2>/dev/null || true)
  if [[ -n "$log_file" ]]; then
    total=$(kv_cache_total_tokens_from_log "$log_file" 2>/dev/null || true)
  fi
  ratio=$(kv_cache_usage_ratio_from_metrics "$api_root" 2>/dev/null || true)

  if [[ -n "$total" && -n "$ratio" ]]; then
    read -r used pct < <(
      awk -v total="$total" -v ratio="$ratio" 'BEGIN {
        r = ratio + 0
        if (r > 1) {
          r = r / 100
        }
        printf "%.0f %.1f\n", total * r, r * 100
      }'
    )
    printf '  %-*s %s used | %s total tokens (%s%%)\n' "$label_width" "Cache:" "$used" "$total" "$pct"
  elif [[ -n "$total" ]]; then
    printf '  %-*s %s total tokens\n' "$label_width" "Cache:" "$total"
  elif [[ -n "$ratio" ]]; then
    pct=$(awk -v ratio="$ratio" 'BEGIN { r = ratio + 0; if (r <= 1) r *= 100; printf "%.1f", r }')
    printf '  %-*s %s%% used\n' "$label_width" "Cache:" "$pct"
  fi
}

service_has_live_kv_cache_usage() {
  local info pid_file pid name api_root ratio

  info=$(current_service_info) || return 1
  IFS=$'\t' read -r pid_file pid name <<< "$info"
  api_root=$(service_api_root "$pid_file" "$pid")
  ratio=$(kv_cache_usage_ratio_from_metrics "$api_root" 2>/dev/null || true)
  [[ -n "$ratio" ]]
}

render_service_status() {
  local info pid_file pid name api

  echo "Service status"
  if info=$(current_service_info); then
    IFS=$'\t' read -r pid_file pid name <<< "$info"
    api=${LAST_API_LAN:-${LAST_API_LOCAL:-http://127.0.0.1:${PORT:-8000}/v1}}
    printf '  Status:  RUNNING\n'
    printf '  Model:   %s\n' "${SERVED_NAME:-$name}"
    printf '  API:     %s\n' "$api"
    printf '  PID:     %s\n' "$pid"
    printf '  Prefix:  %s\n' "$(pid_prefix_cache_label "$pid")"
    printf '  Prompt details: %s\n' "$(pid_prompt_details_label "$pid")"
    render_kv_cache_status "$pid_file" "$pid" "$name" 8
  else
    printf '  Status:  STOPPED\n'
  fi
  echo
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_tty() {
  [[ -t 0 && -t 1 ]] || { : </dev/tty >/dev/tty; } 2>/dev/null
}

pause_enter() {
  is_tty || return 0
  read -r -p "Press Enter to continue..." _
}

normalize_bool() {
  case "${1,,}" in
    1|yes|y|true|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

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

ROUTE_PROFILE_KEYS=(
  SERVED_NAME
  COMPATIBLE_MODES
  MODEL_FAMILY
  PROFILE_GROUP
  MODEL_VARIANT
  QUANTIZATION
  KV_CACHE_DTYPE
  MAX_MODEL_LEN
  KV_CACHE_MEMORY_BYTES
  GPU_UTIL
  MAX_BATCHED_TOKENS
  MAX_NUM_SEQS
  LONG_PREFILL_TOKEN_THRESHOLD
  PREFILL_BATCH_BARRIER
  DISABLE_PREFIX_CACHING
  MTP_K
  MESSAGE_TYPE
  MM_LIMIT_JSON
  LANGUAGE_MODEL_ONLY
  SKIP_MM_PROFILING
  HF_OVERRIDES_JSON
  ADDITIONAL_CONFIG_JSON
  SPECULATIVE_CONFIG
  COMPILATION_CONFIG_JSON
  ATTENTION_BACKEND
  DISABLE_HYBRID_KV_CACHE_MANAGER
  DISABLE_CUSTOM_ALL_REDUCE
  VLLM_ALLOW_LONG_MAX_MODEL_LEN
  VLLM_FORCE_NVFP4_W4A16
  VLLM_PLE_CPU_OFFLOAD
  VLLM_INT8KV_FA_CASCADE_DEQUANT
  VLLM_INT8KV_FA_CASCADE_TILE_TOKENS
  VLLM_INT8KV_FA_CONTINUATION_DEQUANT
  VLLM_INT8KV_FA_PREFILL
  VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE
  VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS
  VLLM_TURBOQUANT_MAX_KV_SPLITS
  VLLM_TURBOQUANT_DECODE_BLOCK_KV
)

NON_INTERACTIVE_CONFIG_KEYS=(
  MODEL_DIR
  PROFILE_DIR
  PROFILE
  PROFILE_FILE
  MODE
  PORT
  SERVICE_SCOPE
  GPU_DEVICES
  TP_SIZE
  PP_SIZE
  CHAT_TEMPLATE_FILE
  CHAT_TEMPLATE_PRESET
  TEMPLATE_DIR
  REASONING_PARSER
  DEFAULT_CHAT_TEMPLATE_KWARGS
  REASONING_MODE
  REASONING_BUDGET
  ENABLE_AUTO_TOOL_CHOICE
  TOOL_CALL_PARSER
  TOOL_PARSER_PLUGIN
  ENABLE_PREFIX_CACHING
  ENABLE_PROMPT_TOKENS_DETAILS
  DISABLE_PREFIX_CACHING
  VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH
  VLLM_ENFORCE_STRICT_TOOL_CALLING
  MAMBA_CACHE_MODE
  ENFORCE_EAGER
  NO_ASYNC_SCHEDULING
  DISABLE_LOG_STATS
  VLLM_SM75_SPEC_SYNC_MODE
  RUNTIME_ROOT
  LOG_DIR
  STATE_FILE
  FLASHQLA_ROOT
  START_TIMEOUT
)

NON_INTERACTIVE_BOOLEAN_KEYS=(
  LANGUAGE_MODEL_ONLY
  SKIP_MM_PROFILING
  ENABLE_AUTO_TOOL_CHOICE
  ENABLE_PREFIX_CACHING
  ENABLE_PROMPT_TOKENS_DETAILS
  DISABLE_PREFIX_CACHING
  VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH
  VLLM_ENFORCE_STRICT_TOOL_CALLING
  ENFORCE_EAGER
  NO_ASYNC_SCHEDULING
  DISABLE_HYBRID_KV_CACHE_MANAGER
  DISABLE_CUSTOM_ALL_REDUCE
  DISABLE_LOG_STATS
  VLLM_ALLOW_LONG_MAX_MODEL_LEN
  VLLM_INT8KV_FA_CASCADE_DEQUANT
  VLLM_INT8KV_FA_CONTINUATION_DEQUANT
  VLLM_INT8KV_FA_PREFILL
)

declare -A CONFIG_FLAG_TO_KEY=()
declare -A CONFIG_KNOWN_KEYS=()
declare -A CONFIG_BOOLEAN_KEYS=()
declare -A CONFIG_OVERRIDE_SOURCE=()
declare -A CONFIG_OVERRIDE_UNSET=()
CONFIG_REGISTRY_INITIALIZED=0

config_key_to_flag() {
  local key=${1,,}
  key=${key//_/-}
  printf -- '--%s\n' "$key"
}

normalize_config_key() {
  local raw=${1#--}
  local key=${raw//-/_}
  key=${key^^}
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "Invalid config key: $1"
  printf '%s\n' "$key"
}

init_config_registry() {
  [[ "$CONFIG_REGISTRY_INITIALIZED" == "1" ]] && return 0

  local key
  for key in "${ROUTE_PROFILE_KEYS[@]}" "${NON_INTERACTIVE_CONFIG_KEYS[@]}"; do
    [[ -n "$key" ]] || continue
    CONFIG_KNOWN_KEYS["$key"]=1
    CONFIG_FLAG_TO_KEY["$(config_key_to_flag "$key")"]="$key"
  done
  for key in "${NON_INTERACTIVE_BOOLEAN_KEYS[@]}"; do
    [[ -n "$key" ]] || continue
    CONFIG_BOOLEAN_KEYS["$key"]=1
  done
  CONFIG_REGISTRY_INITIALIZED=1
}

config_key_has_override() {
  local key=$1
  [[ -n "${CONFIG_OVERRIDE_SOURCE[$key]+x}" ]]
}

config_key_has_explicit_value() {
  local key=$1
  [[ -n "${CONFIG_OVERRIDE_SOURCE[$key]+x}" \
    && "${CONFIG_OVERRIDE_SOURCE[$key]}" != mode \
    && -z "${CONFIG_OVERRIDE_UNSET[$key]+x}" ]]
}

config_key_is_boolean() {
  local key=$1
  [[ -n "${CONFIG_BOOLEAN_KEYS[$key]+x}" ]]
}

env_var_is_exported() {
  printenv "$1" >/dev/null 2>&1
}

set_config_override() {
  local key=$1
  local value=$2
  local source=${3:-cli}

  printf -v "$key" '%s' "$value"
  export "$key"
  CONFIG_OVERRIDE_SOURCE["$key"]="$source"
  unset "CONFIG_OVERRIDE_UNSET[$key]"
}

unset_config_override() {
  local key=$1
  local source=${2:-cli}

  unset "$key"
  CONFIG_OVERRIDE_SOURCE["$key"]="$source"
  CONFIG_OVERRIDE_UNSET["$key"]=1
}

config_key_from_flag() {
  local flag=$1
  case "$flag" in
    --pp-size|--pipeline-parallel-size)
      printf 'PP_SIZE\n'
      return 0
      ;;
  esac
  local key=${CONFIG_FLAG_TO_KEY[$flag]:-}
  [[ -n "$key" ]] || return 1
  printf '%s\n' "$key"
}

parse_set_assignment() {
  local assignment=$1
  local source=${2:-cli}
  local key value

  [[ "$assignment" == *=* ]] || die "--set expects KEY=VALUE."
  key=$(normalize_config_key "${assignment%%=*}")
  value=${assignment#*=}
  set_config_override "$key" "$value" "$source"
}

register_env_config_overrides() {
  init_config_registry

  if ! config_key_has_override GPU_DEVICES && env_var_is_exported CUDA_VISIBLE_DEVICES; then
    set_config_override GPU_DEVICES "${CUDA_VISIBLE_DEVICES:-}" env
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      CONFIG_OVERRIDE_UNSET["GPU_DEVICES"]=1
    fi
  fi

  local key
  for key in "${!CONFIG_KNOWN_KEYS[@]}"; do
    env_var_is_exported "$key" || continue
    config_key_has_override "$key" && continue
    CONFIG_OVERRIDE_SOURCE["$key"]=env
    if [[ -z "${!key:-}" ]]; then
      CONFIG_OVERRIDE_UNSET["$key"]=1
    fi
  done
}

apply_launcher_path_defaults() {
  RUNTIME_ROOT=${RUNTIME_ROOT:-"$MANAGER_ROOT"}
  PROFILE_DIR=${PROFILE_DIR:-"$MANAGER_ROOT/profiles"}
  LOG_DIR=${LOG_DIR:-"$MANAGER_ROOT/run-logs"}

  if config_key_has_override PROFILE_DIR && ! config_key_has_explicit_value TEMPLATE_DIR; then
    TEMPLATE_DIR="$PROFILE_DIR/templates"
  fi
  TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}

  if config_key_has_override LOG_DIR && ! config_key_has_explicit_value STATE_FILE; then
    STATE_FILE="$LOG_DIR/start-manager.state"
  fi
  STATE_FILE=${STATE_FILE:-"$LOG_DIR/start-manager.state"}
}

parse_launcher_args() {
  init_config_registry

  local arg key value
  while (($#)); do
    arg=$1
    shift
    case "$arg" in
      --non-interactive)
        NON_INTERACTIVE=1
        ;;
      --print-config)
        PRINT_CONFIG=1
        NON_INTERACTIVE=1
        ;;
      --set)
        (($#)) || die "--set expects KEY=VALUE."
        parse_set_assignment "$1" cli
        shift
        NON_INTERACTIVE=1
        ;;
      --set=*)
        parse_set_assignment "${arg#--set=}" cli
        NON_INTERACTIVE=1
        ;;
      --unset)
        (($#)) || die "--unset expects KEY."
        key=$(normalize_config_key "$1")
        unset_config_override "$key" cli
        shift
        NON_INTERACTIVE=1
        ;;
      --unset=*)
        key=$(normalize_config_key "${arg#--unset=}")
        unset_config_override "$key" cli
        NON_INTERACTIVE=1
        ;;
      --*=*)
        key=$(config_key_from_flag "${arg%%=*}") || die "Unknown option: ${arg%%=*}"
        value=${arg#*=}
        set_config_override "$key" "$value" cli
        NON_INTERACTIVE=1
        ;;
      --*)
        key=$(config_key_from_flag "$arg") || die "Unknown option: $arg"
        if config_key_is_boolean "$key" && { (($# == 0)) || [[ "${1:-}" == --* ]]; }; then
          value=1
        else
          (($#)) || die "Option $arg expects a value."
          value=$1
          shift
        fi
        set_config_override "$key" "$value" cli
        NON_INTERACTIVE=1
        ;;
      *)
        die "Unknown positional argument: $arg"
        ;;
    esac
  done
}

reset_route_profile_fields() {
  local key
  for key in "${ROUTE_PROFILE_KEYS[@]}"; do
    config_key_has_override "$key" && continue
    unset "$key"
  done
}

profile_key_is_global() {
  case "$1" in
MODEL_DIR|PROFILE_DIR|PROFILE|MODE|PORT|SERVICE_SCOPE|GPU_DEVICES|TP_SIZE|PP_SIZE|\
CHAT_TEMPLATE_FILE|CHAT_TEMPLATE_PRESET|TEMPLATE_DIR|REASONING_PARSER|\
DEFAULT_CHAT_TEMPLATE_KWARGS|REASONING_MODE|REASONING_BUDGET|\
ENABLE_AUTO_TOOL_CHOICE|TOOL_CALL_PARSER|TOOL_PARSER_PLUGIN|\
ENABLE_PREFIX_CACHING|ENABLE_PROMPT_TOKENS_DETAILS|\
VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH|VLLM_ENFORCE_STRICT_TOOL_CALLING)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

source_profile_defaults() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    profile_key_is_global "$key" && continue
    if config_key_has_override "$key" || [[ ${!key+x} ]]; then
      continue
    fi
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

apply_profile_overrides() {
  local file=$1
  [[ -f "$file" ]] || return 0

  local key value
  local preserve_route_env_keys=(
    KV_CACHE_MEMORY_BYTES
    VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE
    VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS
    VLLM_TURBOQUANT_MAX_KV_SPLITS
    VLLM_TURBOQUANT_DECODE_BLOCK_KV
  )
  local preserved_keys=()
  local preserved_values=()
  local preserve_key
  for preserve_key in "${preserve_route_env_keys[@]}"; do
    if [[ -z "$(read_profile_value "$file" "$preserve_key")" && ${!preserve_key+x} ]]; then
      preserved_keys+=("$preserve_key")
      preserved_values+=("${!preserve_key}")
    fi
  done

  reset_route_profile_fields
  local preserve_index
  for preserve_index in "${!preserved_keys[@]}"; do
    printf -v "${preserved_keys[$preserve_index]}" '%s' "${preserved_values[$preserve_index]}"
    export "${preserved_keys[$preserve_index]}"
  done
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    profile_key_is_global "$key" && continue
    config_key_has_override "$key" && continue
    value=$(read_profile_value "$file" "$key")
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' "$file" | sort -u)
}

resolve_profile_file() {
  if [[ -n "${PROFILE_FILE:-}" ]]; then
    printf '%s\n' "$PROFILE_FILE"
    return 0
  fi
  if [[ -z "${PROFILE:-}" ]]; then
    return 1
  fi
  if [[ -f "$PROFILE_DIR/$PROFILE" ]]; then
    printf '%s\n' "$PROFILE_DIR/$PROFILE"
    return 0
  fi
  if [[ -f "$PROFILE_DIR/${PROFILE%.env}.env" ]]; then
    printf '%s\n' "$PROFILE_DIR/${PROFILE%.env}.env"
    return 0
  fi
  return 1
}

load_manager_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

save_manager_state() {
  mkdir -p "$LOG_DIR"
  {
    printf 'MODEL_DIR=%q\n' "${MODEL_DIR:-}"
    printf 'PROFILE_DIR=%q\n' "${PROFILE_DIR:-}"
    printf 'TEMPLATE_DIR=%q\n' "${TEMPLATE_DIR:-}"
    printf 'PROFILE=%q\n' "${PROFILE:-}"
    printf 'MODEL_FAMILY=%q\n' "${MODEL_FAMILY:-}"
    printf 'PROFILE_GROUP=%q\n' "${PROFILE_GROUP:-}"
    printf 'MODEL_VARIANT=%q\n' "${MODEL_VARIANT:-}"
    printf 'SERVED_NAME=%q\n' "${SERVED_NAME:-}"
    printf 'GPU_DEVICES=%q\n' "${GPU_DEVICES:-}"
    printf 'QUANTIZATION=%q\n' "${QUANTIZATION:-}"
    printf 'KV_CACHE_DTYPE=%q\n' "${KV_CACHE_DTYPE:-}"
    printf 'MAMBA_CACHE_MODE=%q\n' "${MAMBA_CACHE_MODE:-}"
    printf 'ENABLE_PREFIX_CACHING=%q\n' "${ENABLE_PREFIX_CACHING:-1}"
    printf 'ENABLE_PROMPT_TOKENS_DETAILS=%q\n' "${ENABLE_PROMPT_TOKENS_DETAILS:-1}"
    printf 'MAX_MODEL_LEN=%q\n' "${MAX_MODEL_LEN:-}"
    printf 'KV_CACHE_MEMORY_BYTES=%q\n' "${KV_CACHE_MEMORY_BYTES:-}"
    printf 'GPU_UTIL=%q\n' "${GPU_UTIL:-}"
    printf 'MAX_BATCHED_TOKENS=%q\n' "${MAX_BATCHED_TOKENS:-}"
    printf 'MAX_NUM_SEQS=%q\n' "${MAX_NUM_SEQS:-}"
    printf 'LONG_PREFILL_TOKEN_THRESHOLD=%q\n' "${LONG_PREFILL_TOKEN_THRESHOLD:-}"
    printf 'PREFILL_BATCH_BARRIER=%q\n' "${PREFILL_BATCH_BARRIER:-0}"
    printf 'DISABLE_PREFIX_CACHING=%q\n' "${DISABLE_PREFIX_CACHING:-0}"
    printf 'MTP_K=%q\n' "${MTP_K:-}"
    printf 'VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=%q\n' "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-}"
    printf 'VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS=%q\n' "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-}"
    printf 'VLLM_TURBOQUANT_MAX_KV_SPLITS=%q\n' "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-}"
    printf 'VLLM_TURBOQUANT_DECODE_BLOCK_KV=%q\n' "${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-}"
    printf 'MESSAGE_TYPE=%q\n' "${MESSAGE_TYPE:-}"
    printf 'MM_LIMIT_JSON=%q\n' "${MM_LIMIT_JSON:-}"
    printf 'LANGUAGE_MODEL_ONLY=%q\n' "${LANGUAGE_MODEL_ONLY:-}"
    printf 'SKIP_MM_PROFILING=%q\n' "${SKIP_MM_PROFILING:-}"
    printf 'HF_OVERRIDES_JSON=%q\n' "${HF_OVERRIDES_JSON:-}"
    printf 'ADDITIONAL_CONFIG_JSON=%q\n' "${ADDITIONAL_CONFIG_JSON:-}"
    printf 'SPECULATIVE_CONFIG=%q\n' "${SPECULATIVE_CONFIG:-}"
    printf 'COMPILATION_CONFIG_JSON=%q\n' "${COMPILATION_CONFIG_JSON:-}"
    printf 'TP_SIZE=%q\n' "${TP_SIZE:-}"
    printf 'PP_SIZE=%q\n' "${PP_SIZE:-1}"
    printf 'CHAT_TEMPLATE_FILE=%q\n' "${CHAT_TEMPLATE_FILE:-}"
    printf 'CHAT_TEMPLATE_PRESET=%q\n' "${CHAT_TEMPLATE_PRESET:-}"
    printf 'ATTENTION_BACKEND=%q\n' "${ATTENTION_BACKEND:-}"
    printf 'REASONING_MODE=%q\n' "${REASONING_MODE:-}"
    printf 'REASONING_PARSER=%q\n' "${REASONING_PARSER:-}"
    printf 'REASONING_BUDGET=%q\n' "${REASONING_BUDGET:-}"
    printf 'DEFAULT_CHAT_TEMPLATE_KWARGS=%q\n' "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
    printf 'ENABLE_AUTO_TOOL_CHOICE=%q\n' "${ENABLE_AUTO_TOOL_CHOICE:-0}"
    printf 'TOOL_CALL_PARSER=%q\n' "${TOOL_CALL_PARSER:-}"
    printf 'TOOL_PARSER_PLUGIN=%q\n' "${TOOL_PARSER_PLUGIN:-}"
    printf 'VLLM_ENFORCE_STRICT_TOOL_CALLING=%q\n' "${VLLM_ENFORCE_STRICT_TOOL_CALLING:-}"
    printf 'ENFORCE_EAGER=%q\n' "${ENFORCE_EAGER:-}"
    printf 'NO_ASYNC_SCHEDULING=%q\n' "${NO_ASYNC_SCHEDULING:-}"
    printf 'DISABLE_HYBRID_KV_CACHE_MANAGER=%q\n' "${DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
    printf 'DISABLE_PREFIX_CACHING=%q\n' "${DISABLE_PREFIX_CACHING:-}"
    printf 'DISABLE_CUSTOM_ALL_REDUCE=%q\n' "${DISABLE_CUSTOM_ALL_REDUCE:-}"
    printf 'DISABLE_LOG_STATS=%q\n' "${DISABLE_LOG_STATS:-}"
    printf 'VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=%q\n' "${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-}"
    printf 'MODE=%q\n' "${MODE:-normal}"
    printf 'PORT=%q\n' "${PORT:-8000}"
    printf 'SERVICE_SCOPE=%q\n' "${SERVICE_SCOPE:-local}"
    printf 'LAST_PID_FILE=%q\n' "${LAST_PID_FILE:-}"
    printf 'LAST_LOG_FILE=%q\n' "${LAST_LOG_FILE:-}"
    printf 'LAST_API_LOCAL=%q\n' "${LAST_API_LOCAL:-}"
    printf 'LAST_API_LAN=%q\n' "${LAST_API_LAN:-}"
    printf 'LAST_SMOKE_OUTPUT=%q\n' "${LAST_SMOKE_OUTPUT:-}"
  } > "$STATE_FILE"
}

list_profiles() {
  [[ -d "$PROFILE_DIR" ]] || return 0
  find "$PROFILE_DIR" -type f -name '*.env' -printf '%P\n' | sort
}

list_profiles_for_model() {
  local family=$1
  local quantization=${2:-}
  local profile profile_file profile_family profile_variant

  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    profile_file="$PROFILE_DIR/$profile"
    profile_family=$(read_profile_value "$profile_file" MODEL_FAMILY)
    profile_variant=$(read_profile_value "$profile_file" MODEL_VARIANT)

    if [[ "$family" == qwen* && -n "$profile_family" && "$profile_family" != qwen* ]]; then
      continue
    fi
    if [[ "$family" == gemma* && -n "$profile_family" && "$profile_family" != gemma* ]]; then
      continue
    fi

    if [[ "$family" == qwen* && "$quantization" == fp8 && "$profile_variant" != fp8 ]]; then
      continue
    fi
    if [[ "$family" == qwen* && -n "$quantization" && "$quantization" != fp8 && "$profile_variant" == fp8 ]]; then
      continue
    fi

    printf '%s\n' "$profile"
  done < <(list_profiles)
}

first_compatible_mode() {
  local compatible_modes=${1:-safe,normal,fast}
  local candidate
  for candidate in ${compatible_modes//,/ }; do
    candidate=${candidate//[[:space:]]/}
    case "$candidate" in
      stable) echo safe; return 0 ;;
      speed) echo normal; return 0 ;;
      safe|normal|fast|aggressive) echo "$candidate"; return 0 ;;
    esac
  done
  echo safe
}

mode_is_compatible() {
  local mode=$1
  local compatible_modes=${2:-safe,normal,fast}
  local candidate

  for candidate in ${compatible_modes//,/ }; do
    candidate=${candidate//[[:space:]]/}
    case "$candidate" in
      stable) candidate=safe ;;
      speed) candidate=normal ;;
    esac
    if [[ "$mode" == "aggressive" ]]; then
      [[ "$candidate" == "aggressive" || "$candidate" == "fast" ]] && return 0
      continue
    fi
    [[ "$candidate" == "$mode" ]] && return 0
  done
  return 1
}

profile_family_dir() {
  if [[ -n "${PROFILE:-}" && "$PROFILE" == */* ]]; then
    printf '%s\n' "${PROFILE%%/*}"
    return 0
  fi
  case "${MODEL_FAMILY:-}" in
    gemma*) echo gemma31b ;;
    qwen*|"") echo qwen27b ;;
    *)
      printf '%s\n' "${MODEL_FAMILY//[^A-Za-z0-9_.-]/-}"
      ;;
  esac
}

profile_weight_dir() {
  local variant=${MODEL_VARIANT:-}
  local precision

  case "${variant,,}" in
    fp8|w8a16)
      printf 'w8a16\n'
      return 0
      ;;
    nvfp4|w4a16|int4|gptq|awq)
      printf 'w4a16\n'
      return 0
      ;;
  esac

  precision=$(guess_precision_scheme "${MODEL_DIR:-}" "${QUANTIZATION:-}")
  case "$precision" in
    W4A16) printf 'w4a16\n' ;;
    *) printf 'w8a16\n' ;;
  esac
}

profile_compatible_modes_for_current() {
  normalize_mode
  local mode=${MODE:-normal}
  local kv=${KV_CACHE_DTYPE:-}
  local mtp=${MTP_K:-0}

  if [[ "$mode" == "aggressive" ]]; then
    echo aggressive
    return 0
  fi

  if [[ "$mode" == "safe" ]]; then
    case "$kv" in
      ""|fp16|default|auto)
        ;;
      *)
        if [[ "$mtp" =~ ^[0-9]+$ ]] && (( mtp > 0 )); then
          echo fast
          return 0
        fi
        ;;
    esac
  fi
  echo "$mode"
}

sanitize_profile_name() {
  local name=$1
  name=${name%.env}
  name=$(printf '%s' "$name" | sed -E 's/[^A-Za-z0-9_.-]+/-/g; s/^-+//; s/-+$//')
  [[ -n "$name" ]] || name="user-profile"
  printf '%s\n' "$name"
}

write_profile_entry() {
  local file=$1
  local key=$2
  local value=${3:-}
  [[ -n "$value" ]] || return 0
  value=${value//\'/}
  printf "%s='%s'\n" "$key" "$value" >> "$file"
}

list_template_presets() {
  [[ -d "$TEMPLATE_DIR" ]] || return 0
  find "$TEMPLATE_DIR" -type f \( -name '*.jinja' -o -name '*.jinja2' -o -name '*.txt' \) -printf '%P\n' |
    sort
}

resolve_template_file() {
  local template=${1:-}
  [[ -n "$template" ]] || return 1
  if [[ -f "$template" ]]; then
    printf '%s\n' "$template"
    return 0
  fi
  if [[ -f "$TEMPLATE_DIR/$template" ]]; then
    printf '%s\n' "$TEMPLATE_DIR/$template"
    return 0
  fi
  return 1
}

current_template_label() {
  if [[ -n "${CHAT_TEMPLATE_PRESET:-}" ]]; then
    printf '%s\n' "$CHAT_TEMPLATE_PRESET"
  elif [[ -n "${CHAT_TEMPLATE_FILE:-}" ]]; then
    printf '%s\n' "$CHAT_TEMPLATE_FILE"
  else
    printf 'model default'
  fi
}

current_reasoning_label() {
  local label="template default"
  if [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    case "${DEFAULT_CHAT_TEMPLATE_KWARGS//[[:space:]]/}" in
      *'"enable_thinking":false'*|*"\"enable_thinking\":false"*)
        label="thinking off"
        ;;
      *'"enable_thinking":true'*|*"\"enable_thinking\":true"*)
        label="thinking on"
        ;;
      *)
        label="custom kwargs"
        ;;
    esac
  fi
  if [[ -n "${REASONING_PARSER:-}" ]]; then
    label+=" / parser=$REASONING_PARSER"
  fi
  if [[ -n "${REASONING_BUDGET:-}" ]]; then
    label+=" / default budget=$REASONING_BUDGET"
  fi
  printf '%s\n' "$label"
}

reasoning_parser_is_disabled() {
  case "${REASONING_PARSER:-}" in
    off|none|disabled|disable)
      return 0
      ;;
  esac
  return 1
}

default_qwen_reasoning_parser_applies() {
  local model_dir_l served_l profile_l group_l

  [[ "${MODEL_FAMILY:-}" == qwen* ]] || return 1

  model_dir_l=${MODEL_DIR,,}
  served_l=${SERVED_NAME,,}
  profile_l=${PROFILE:-}
  profile_l=${profile_l,,}
  group_l=${PROFILE_GROUP:-}
  group_l=${group_l,,}

  case "$group_l" in
    qwen3*|qwen36*)
      return 0
      ;;
  esac
  case "$profile_l" in
    qwen27b/*)
      return 0
      ;;
  esac
  case "$model_dir_l $served_l" in
    *qwen3*|*qwen-3*|*qwen_3*|*qwopus3*|*qwen36*)
      return 0
      ;;
  esac
  return 1
}

apply_family_reasoning_defaults() {
  # Qwen3/Qwen3.5 tokenizer configs do not always advertise the parser.
  # Keep request thinking defaults template-driven, but make response parsing
  # explicit so thinking text is not returned as normal content.
  if config_key_has_explicit_value REASONING_PARSER; then
    return 0
  fi
  if reasoning_parser_is_disabled; then
    return 0
  fi
  if default_qwen_reasoning_parser_applies; then
    REASONING_PARSER=${REASONING_PARSER:-qwen3}
  elif [[ "${REASONING_PARSER:-}" == "qwen3" ]]; then
    REASONING_PARSER=""
  fi
}

apply_prefix_cache_defaults() {
  ENABLE_PREFIX_CACHING=$(normalize_bool "${ENABLE_PREFIX_CACHING:-1}")
  ENABLE_PROMPT_TOKENS_DETAILS=$(normalize_bool "${ENABLE_PROMPT_TOKENS_DETAILS:-1}")
  DISABLE_PREFIX_CACHING=$(normalize_bool "${DISABLE_PREFIX_CACHING:-0}")

  if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
    ENABLE_PREFIX_CACHING=0
    return 0
  fi

  if config_key_has_explicit_value MAMBA_CACHE_MODE; then
    return 0
  fi

  if [[ "$ENABLE_PREFIX_CACHING" == "1" && "$MODEL_FAMILY" == qwen* ]]; then
    MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-align}
  elif [[ "${MAMBA_CACHE_MODE:-}" == "align" ]]; then
    # align is only injected as the Qwen prefix-cache default. Clear it when
    # the current route no longer uses that default to avoid stale state bleed.
    MAMBA_CACHE_MODE=""
  fi
}

normalize_message_type_defaults() {
  if config_key_has_explicit_value MESSAGE_TYPE; then
    MESSAGE_TYPE=${MESSAGE_TYPE:-text-only}
  elif [[ "${MESSAGE_TYPE:-}" == "text+image" || -n "${MM_LIMIT_JSON:-}" || "${LANGUAGE_MODEL_ONLY:-1}" == "0" ]]; then
    MESSAGE_TYPE=text+image
  else
    MESSAGE_TYPE=text-only
  fi

  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    if ! config_key_has_explicit_value MM_LIMIT_JSON; then
      MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    fi
    if ! config_key_has_explicit_value LANGUAGE_MODEL_ONLY; then
      LANGUAGE_MODEL_ONLY=0
    fi
    if ! config_key_has_explicit_value SKIP_MM_PROFILING; then
      SKIP_MM_PROFILING=$(normalize_bool "${SKIP_MM_PROFILING:-0}")
    fi
  else
    if ! config_key_has_explicit_value MM_LIMIT_JSON; then
      MM_LIMIT_JSON=""
    fi
    if ! config_key_has_explicit_value LANGUAGE_MODEL_ONLY; then
      LANGUAGE_MODEL_ONLY=1
    fi
    if ! config_key_has_explicit_value SKIP_MM_PROFILING; then
      SKIP_MM_PROFILING=1
    fi
  fi
}

current_tool_calling_label() {
  local label plugin_label

  if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
    label="auto"
    if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
      label+=" / parser=$TOOL_CALL_PARSER"
    else
      label+=" / parser=<unset>"
    fi
  else
    label="off"
    if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
      label+=" / parser=$TOOL_CALL_PARSER"
    fi
  fi

  if [[ -n "${TOOL_PARSER_PLUGIN:-}" ]]; then
    plugin_label=${TOOL_PARSER_PLUGIN##*/}
    label+=" / plugin=$plugin_label"
  fi

  printf '%s\n' "$label"
}

current_prefix_cache_label() {
  if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    printf 'disabled'
  elif [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    printf 'enabled'
  else
    printf 'auto'
  fi
}

current_tq_diagnostics_label() {
  if [[ "${KV_CACHE_DTYPE:-}" != turboquant_* ]]; then
    printf 'n/a'
    return 0
  fi
  printf 'FORCE_DECODE_SDPA=%s, FORCE_CONTINUATION_SDPA=%s, PREFIX_COMBINE=%s@%s, MAX_KV_SPLITS=%s, DECODE_BLOCK_KV=%s, K8V4_FP8_FORMAT=%s' \
    "${VLLM_TURBOQUANT_FORCE_DECODE_SDPA:-0}" \
    "${VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA:-0}" \
    "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-auto}" \
    "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-20480}" \
    "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-auto}" \
    "${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-4}" \
    "${VLLM_TURBOQUANT_K8V4_FP8_FORMAT:-auto}"
}

gpu_device_count() {
  local devices=${1:-}
  local count=0 part
  devices=${devices// /}
  [[ -n "$devices" ]] || {
    echo 0
    return 0
  }
  IFS=',' read -r -a parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

parallelism_options() {
  local gpu_count=$1 pp tp
  [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || return 0
  for ((pp = 1; pp <= gpu_count; pp++)); do
    (( gpu_count % pp == 0 )) || continue
    tp=$((gpu_count / pp))
    printf 'TP%s / PP%s\n' "$tp" "$pp"
  done
}

select_parallelism_menu() {
  local gpu_count=$1 options=() default selected
  mapfile -t options < <(parallelism_options "$gpu_count")
  ((${#options[@]} > 0)) || return 1
  default="TP${TP_SIZE:-$gpu_count} / PP${PP_SIZE:-1}"
  printf '%s\n' "${options[@]}" | grep -Fxq "$default" || default="TP${gpu_count} / PP1"
  selected=$(menu_select "Tensor / pipeline parallelism ($gpu_count GPU(s))" "$default" "${options[@]}") || return 1
  [[ "$selected" =~ ^TP([0-9]+)[[:space:]]*/[[:space:]]*PP([0-9]+)$ ]] || return 1
  TP_SIZE=${BASH_REMATCH[1]}
  PP_SIZE=${BASH_REMATCH[2]}
}

validate_parallelism() {
  local gpu_count expected
  gpu_count=$(gpu_device_count "${GPU_DEVICES:-}")
  if [[ ! "${TP_SIZE:-}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${PP_SIZE:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TP_SIZE and PP_SIZE must be positive integers." >&2
    return 1
  fi
  expected=$((TP_SIZE * PP_SIZE))
  if (( gpu_count != expected )); then
    echo "ERROR: selected GPU count ($gpu_count) must equal TP_SIZE ($TP_SIZE) * PP_SIZE ($PP_SIZE) = $expected." >&2
    return 1
  fi
}

enforce_parallelism_constraints() {
  local pp=${PP_SIZE:-1}
  local mtp=${MTP_K:-0}

  # vLLM's PP async path currently accepts one sampled token per request.
  # MTP/rejection sampling returns multiple tokens, so PP + MTP must use the
  # synchronous scheduler until upstream supports that state transfer.
  if [[ "$pp" =~ ^[2-9][0-9]*$ ]] && [[ "$mtp" =~ ^[1-9][0-9]*$ ]]; then
    if [[ "${NO_ASYNC_SCHEDULING:-0}" != "1" ]]; then
      NO_ASYNC_SCHEDULING=1
      PP_MTP_ASYNC_AUTO_DISABLED=1
    fi
  fi

  # PLE CPU offload has one shared request/output slot per DP rank. The PP
  # async scheduler can advance a later microstep before that slot's reset is
  # visible to the CPU worker, leaving the pipeline waiting indefinitely.
  if [[ "$pp" =~ ^[2-9][0-9]*$ ]] && [[ "${VLLM_PLE_CPU_OFFLOAD:-0}" == "1" || "${VLLM_PLE_CPU_OFFLOAD:-}" == "true" ]]; then
    if [[ "${NO_ASYNC_SCHEDULING:-0}" != "1" ]]; then
      NO_ASYNC_SCHEDULING=1
      PP_PLE_ASYNC_AUTO_DISABLED=1
    fi
  fi
}

list_nvidia_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null |
    awk -F, '
      {
        idx = $1
        name = substr($0, index($0, ",") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", idx)
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (idx != "" && name != "") {
          print idx "\t" name
        }
      }
    '
}

warn_display_gpu_occupancy() {
  local devices=${GPU_DEVICES:-}
  local device pids pid comm args found=0

  command -v fuser >/dev/null 2>&1 || return 0
  devices=${devices// /}
  [[ -n "$devices" ]] || return 0

  IFS=',' read -r -a parts <<< "$devices"
  for device in "${parts[@]}"; do
    [[ "$device" =~ ^[0-9]+$ && -e "/dev/nvidia${device}" ]] || continue
    pids=$(fuser "/dev/nvidia${device}" 2>/dev/null || true)
    [[ -n "$pids" ]] || continue
    for pid in $pids; do
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
      args=$(ps -p "$pid" -o args= 2>/dev/null || true)
      case "$comm $args" in
        *Xorg*|*Xwayland*|*gnome-shell*|*kwin*|*plasmashell*|*Hyprland*|*sway*|*gdm*|*sddm*|*lightdm*)
          if (( found == 0 )); then
            echo
            echo "Display GPU warning"
            echo "  A desktop/display process is using one of the selected compute GPUs."
            echo "  This reduces available VRAM and can lower the maximum stable context."
            echo "  Prefer an iGPU or a non-compute display GPU for large-context profiles."
            found=1
          fi
          printf '  GPU %s: pid=%s %s\n' "$device" "$pid" "${comm:-unknown}"
          ;;
      esac
    done
  done
}

profile_summary() {
  local profile_file=$1
  [[ -f "$profile_file" ]] || return 0

  local keys=(
    SERVED_NAME
    COMPATIBLE_MODES
    MODEL_FAMILY
    PROFILE_GROUP
    MODEL_VARIANT
    QUANTIZATION
    KV_CACHE_DTYPE
    MAX_MODEL_LEN
    KV_CACHE_MEMORY_BYTES
    GPU_UTIL
    MAX_BATCHED_TOKENS
    MAX_NUM_SEQS
    MTP_K
    VLLM_ALLOW_LONG_MAX_MODEL_LEN
    MM_LIMIT_JSON
    HF_OVERRIDES_JSON
  )

  local key value
  for key in "${keys[@]}"; do
    value=$(read_profile_value "$profile_file" "$key")
    [[ -n "$value" ]] || value="-"
    printf '  %-22s %s\n' "$key" "$value"
  done
}

menu_select() {
  local title=$1
  local default=$2
  shift
  shift
  local options=("$@")
  local count=${#options[@]}
  local idx=0
  local key next_key answer answer_rest selected_index number_buffer=""

  (( count > 0 )) || return 1
  for i in "${!options[@]}"; do
    if [[ "${options[$i]}" == "$default" ]]; then
      idx=$i
      break
    fi
  done

  if ! is_tty; then
    printf '%s\n' "${options[$idx]}"
    return 0
  fi

  while true; do
    clear >/dev/tty
    {
      banner
      echo "$title"
      echo
      for i in "${!options[@]}"; do
        if (( i == idx )); then
          printf ' > %d. %s\n' "$((i + 1))" "${options[$i]}"
        else
          printf '   %d. %s\n' "$((i + 1))" "${options[$i]}"
        fi
      done
      echo
      if (( count >= 10 )); then
        echo "Use Up/Down, Enter to select. Number keys select directly. Esc returns."
        [[ -n "$number_buffer" ]] && echo "Input: $number_buffer"
      else
        echo "Press a number to select, Enter for the highlighted item."
      fi
    } >/dev/tty

    if (( count >= 10 )); then
      printf 'Select [1-%s]: ' "$count" >/dev/tty
      IFS= read -rsn1 key </dev/tty || true
      printf '\n' >/dev/tty
      [[ "$key" == $'\x04' ]] && return 1

      if [[ -z "$key" ]]; then
        if [[ -n "$number_buffer" ]]; then
          if [[ "$number_buffer" =~ ^[0-9]+$ ]] && (( number_buffer >= 1 && number_buffer <= count )); then
            printf '%s\n' "${options[$((number_buffer - 1))]}"
            return 0
          fi
          echo "Please enter a listed number." >&2
          number_buffer=""
          sleep 1
          continue
        fi
        printf '%s\n' "${options[$idx]}"
        return 0
      fi

      if [[ "$key" == $'\x1b' ]]; then
        read -rsn2 -t 0.1 key </dev/tty || true
        if [[ -z "$key" ]]; then
          return 1
        fi
        case "$key" in
          "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
          "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
        esac
        continue
      fi

      if [[ "$key" =~ ^[0-9]$ ]]; then
        number_buffer+="$key"
        if [[ "$number_buffer" =~ ^[0-9]+$ ]] \
          && (( 10#$number_buffer >= 1 && 10#$number_buffer <= count )); then
          if menu_index_prefix_exists "$number_buffer" "$count"; then
            next_key=""
            read -rsn1 -t "$MENU_DIGIT_TIMEOUT" next_key </dev/tty || true
            if [[ "$next_key" =~ ^[0-9]$ ]]; then
              number_buffer+="$next_key"
            elif [[ -z "$next_key" ]]; then
              printf '%s\n' "${options[$((10#$number_buffer - 1))]}"
              return 0
            fi
          fi
          if [[ "$number_buffer" =~ ^[0-9]+$ ]] \
            && (( 10#$number_buffer >= 1 && 10#$number_buffer <= count )) \
            && ! menu_index_prefix_exists "$number_buffer" "$count"; then
            printf '%s\n' "${options[$((10#$number_buffer - 1))]}"
            return 0
          fi
        fi
        if ! menu_index_prefix_exists "$number_buffer" "$count"; then
          echo "Please enter a listed number." >&2
          number_buffer=""
          sleep 1
        fi
        continue
      fi

      if [[ "$key" == $'\x7f' || "$key" == $'\b' ]]; then
        number_buffer=${number_buffer%?}
        continue
      fi

      answer="$number_buffer$key"
      number_buffer=""
      read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
      answer="${answer}${answer_rest}"
      for i in "${!options[@]}"; do
        if [[ "${answer,,}" == "${options[$i],,}" ]]; then
          printf '%s\n' "${options[$i]}"
          return 0
        fi
      done

      echo "Please enter a listed number." >&2
      sleep 1
      continue
    fi

    printf 'Select [1-%s]: ' "$count" >/dev/tty
    IFS= read -rsn1 key </dev/tty || true
    printf '\n' >/dev/tty
    if [[ "$key" == $'\x04' ]]; then
      return 1
    fi

    if [[ -z "$key" ]]; then
      printf '%s\n' "${options[$idx]}"
      return 0
    fi

    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      [[ -z "$key" ]] && return 1
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
      continue
    fi

    if [[ "$key" =~ ^[0-9]$ ]]; then
      selected_index="$key"
      if (( count >= 10 && key == 1 )); then
        read -rsn1 -t "$MENU_DIGIT_TIMEOUT" answer </dev/tty || true
        if [[ "$answer" =~ ^[0-9]$ ]]; then
          selected_index="${key}${answer}"
        fi
      fi
      if (( selected_index >= 1 && selected_index <= count )); then
        printf '%s\n' "${options[$((selected_index - 1))]}"
        return 0
      fi
      echo "Please press a listed number." >&2
      sleep 1
      continue
    fi

    answer="$key"
    read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
    answer="${answer}${answer_rest}"
    for i in "${!options[@]}"; do
      if [[ "${answer,,}" == "${options[$i],,}" ]]; then
        printf '%s\n' "${options[$i]}"
        return 0
      fi
    done
    echo "Please press a listed number." >&2
    sleep 1
  done
}

menu_index_prefix_exists() {
  local prefix=$1
  local max=$2
  local i

  [[ "$prefix" =~ ^[0-9]+$ ]] || return 1
  for ((i = 1; i <= max; i++)); do
    [[ "$i" == "$prefix" ]] && continue
    [[ "$i" == "$prefix"* ]] && return 0
  done
  return 1
}

read_menu_key() {
  local max=$1
  local key next selected_index

  IFS= read -rsn1 key </dev/tty || return 1
  printf '\n' >/dev/tty
  [[ -n "$key" ]] || return 1
  [[ "$key" == $'\x04' ]] && return 1
  [[ "$key" == $'\x1b' ]] && return 1

  if [[ "$key" =~ ^[0-9]$ ]]; then
    selected_index="$key"
    if (( max >= 10 && key == 1 )); then
      read -rsn1 -t 0.25 next </dev/tty || true
      if [[ "$next" =~ ^[0-9]$ ]]; then
        selected_index="${key}${next}"
      fi
    fi
    if (( selected_index >= 0 && selected_index <= max )); then
      printf '%s\n' "$selected_index"
      return 0
    fi
  fi

  printf '%s\n' "$key"
}

read_line_with_esc() {
  local prompt=$1
  local key buffer=""

  printf '%s' "$prompt" >/dev/tty
  while true; do
    IFS= read -rsn1 key </dev/tty || return 130
    case "$key" in
      $'\x1b'|$'\x04')
        printf '\n' >/dev/tty
        return 130
        ;;
      "")
        printf '\n' >/dev/tty
        printf '%s\n' "$buffer"
        return 0
        ;;
      $'\x7f'|$'\b')
        if [[ -n "$buffer" ]]; then
          buffer=${buffer%?}
          printf '\b \b' >/dev/tty
        fi
        ;;
      *)
        buffer+="$key"
        printf '%s' "$key" >/dev/tty
        ;;
    esac
  done
}

prompt_default() {
  local label=$1
  local default=$2
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  answer=$(read_line_with_esc "$label [$default] (Esc to cancel): ") || return 130
  if [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_required_dir() {
  local label=$1
  local default=$2
  local answer path

  while true; do
    if [[ -n "$default" ]]; then
      answer=$(read_line_with_esc "$label [$default] (Esc/q to cancel): ") || return 1
    else
      answer=$(read_line_with_esc "$label [required, Esc/q to cancel]: ") || return 1
    fi
    case "${answer,,}" in
      q|quit|exit)
        return 1
        ;;
    esac
    [[ -z "$answer" && -n "$default" ]] && answer="$default"
    [[ -z "$answer" ]] && return 1

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
    echo "Directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_checkpoint_dir() {
  local default=$1
  local answer path

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  while true; do
    if [[ -n "$default" ]]; then
      answer=$(read_line_with_esc "Checkpoint directory [$default] (Esc/q to quit): ") || return 1
    else
      answer=$(read_line_with_esc "Checkpoint directory [required, Esc/q to quit]: ") || return 1
    fi

    case "${answer,,}" in
      q|quit|exit)
        echo "Start cancelled." >&2
        return 1
        ;;
    esac

    if [[ -z "$answer" && -n "$default" ]]; then
      answer="$default"
    fi
    if [[ -z "$answer" ]]; then
      echo "No checkpoint directory selected. Start cancelled." >&2
      return 1
    fi

    path="$answer"
    if [[ "$path" == "~" ]]; then
      path="$HOME"
    elif [[ "$path" == "~/"* ]]; then
      path="$HOME/${path#~/}"
    fi

    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi

    echo "Checkpoint directory does not exist: $path" >&2
    default="$path"
  done
}

prompt_choice() {
  local title=$1
  local current=$2
  shift
  shift
  menu_select "$title" "$current" "$@"
}

prompt_segmented() {
  local title=$1
  local current=$2
  shift
  shift
  local options=("$@")
  local key answer answer_rest i option

  while true; do
    printf '%s:\n' "$title" >/dev/tty
    for i in "${!options[@]}"; do
      option=${options[$i]}
      if [[ "$current" == "$option" ]]; then
        printf '  %d. [x] %s\n' "$((i + 1))" "$option" >/dev/tty
      else
        printf '  %d. [ ] %s\n' "$((i + 1))" "$option" >/dev/tty
      fi
    done
    printf 'Choose 1-%s, value, or Enter to keep: ' "${#options[@]}" >/dev/tty
    IFS= read -rsn1 key </dev/tty || true
    printf '\n' >/dev/tty
    [[ "$key" == $'\x1b' || "$key" == $'\x04' ]] && return 1
    case "${key,,}" in
      "" )
        printf '%s\n' "$current"
        return 0
        ;;
    esac

    if [[ "$key" =~ ^[0-9]$ ]] && (( key >= 1 && key <= ${#options[@]} )); then
      printf '%s\n' "${options[$((key - 1))]}"
      return 0
    fi

    if ((${#options[@]} == 2)); then
      case "${key,,}" in
        left|l)
          printf '%s\n' "${options[0]}"
          return 0
          ;;
        right|r)
          printf '%s\n' "${options[1]}"
          return 0
          ;;
      esac
    fi

    answer="$key"
    read -r -t 0.5 answer_rest </dev/tty || answer_rest=""
    answer="${answer}${answer_rest}"
    for option in "${options[@]}"; do
      if [[ "${answer,,}" == "${option,,}" ]]; then
        printf '%s\n' "$option"
        return 0
      fi
    done

    echo "Please choose a listed number, value, or Enter." >&2
  done
}

confirm_start() {
  local answer

  if ! is_tty; then
    return 0
  fi

  while true; do
    answer=$(read_line_with_esc "Start server now? [y/N]: ") || return 1
    case "$answer" in
      y|Y)
        return 0
        ;;
      n|N|"")
        echo "Start cancelled."
        return 1
        ;;
      *)
        echo "Please type y to start or n to exit."
        ;;
    esac
  done
}

show_help() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  cat <<'EOF'
This is the vLLM 2080 Ti Definitive service manager for a source checkout.

Main menu:
  1. Weight directory: choose the checkpoint directory.
  2. Profile: choose a profile directory, apply .env route presets, select a
     chat-template preset, and edit the filled runtime parameters.
  3. GPU / TP / PP selection: select GPUs with Space, then choose a valid layout.
  4. Launch mode: safe, normal, fast, or aggressive.
  5. Port: default 8000.
  6. Service scope: local only or local + LAN.
  7. Help.
  8. Start service: launch vLLM, wait for /health, run a small smoke request,
     then print API URL, served model, PID file, and log file.
  9. Stop service.
  0. Exit.

Profiles are optional. They are presets only; the current menu values are the
actual launch configuration.

Notes:
  - safe mode: eager fallback. Use it only for diagnosis or conservative
    fallback, not as the formal serving performance route.
  - normal mode: non-eager production default once the selected route has
    passed quality smoke.
  - fast mode: non-eager + full graph. Highest throughput path,
    intended for performance exploration and quality-risk-tolerant use.
  - aggressive mode: more aggressive mode with the highest performance and
    quality risk.
  - Chat-template presets live under profiles/templates and are global launcher
    settings, not route-profile fields.
  - Tool-calling defaults are global launcher settings. Enable automatic tool
    choice only when a matching --tool-call-parser is selected. The launcher
    enables strict tool-output constraints for automatic tool choice.
  - thinking_token_budget is a per-request chat parameter in this vLLM runtime.
  - text+image requires a checkpoint that actually supports vision inputs.
  - Use --set KEY=VALUE for advanced envs such as VLLM_* or compiler paths.
  - Non-interactive runs accept --tp-size together with --pp-size (or
    --pipeline-parallel-size). The selected GPU count must equal TP * PP.
    Qwen3.5 MTP runs as a complete draft layer on the final PP stage. For
    PP>1 + MTP, async scheduling is disabled automatically because the current
    upstream PP async state contract does not support multi-token sampling.
  - Use --unset KEY to clear inherited profile/env values and fall back to
    launcher defaults; use --set KEY= to force an empty value when allowed.
  - --print-config prints the final launch summary and exits without starting.
EOF
  echo
  pause_enter
}

show_profiles() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Profile presets:"
  echo
  local profile profile_file family variant mode kv context kv_memory mtp seqs
  if [[ ! -d "$PROFILE_DIR" ]]; then
    echo "No profile directory found: $PROFILE_DIR"
    echo
    pause_enter
    return 0
  fi
  while IFS= read -r profile; do
    [[ -n "$profile" ]] || continue
    profile_file="$PROFILE_DIR/$profile"
    family=$(read_profile_value "$profile_file" MODEL_FAMILY)
    variant=$(read_profile_value "$profile_file" MODEL_VARIANT)
    mode=$(read_profile_value "$profile_file" COMPATIBLE_MODES)
    [[ -n "$mode" ]] || mode=$(read_profile_value "$profile_file" MODE)
    kv=$(read_profile_value "$profile_file" KV_CACHE_DTYPE)
    context=$(read_profile_value "$profile_file" MAX_MODEL_LEN)
    kv_memory=$(read_profile_value "$profile_file" KV_CACHE_MEMORY_BYTES)
    mtp=$(read_profile_value "$profile_file" MTP_K)
    seqs=$(read_profile_value "$profile_file" MAX_NUM_SEQS)
    printf '  %-62s compatible=%-12s family=%-7s weight=%-6s kv=%-24s ctx=%-8s kvmem=%-10s mtp=%-3s seqs=%s\n' \
      "$profile" "${mode:-safe,normal,fast}" "${family:-auto}" "${variant:-auto}" "${kv:-fp16}" "${context:-auto}" "${kv_memory:-auto}" "${mtp:-0}" "${seqs:-1}"
  done < <(list_profiles)
  echo
  pause_enter
}

current_scope_label() {
  if [[ "${SERVICE_SCOPE:-local}" == "lan" ]]; then
    echo "local + LAN"
  else
    echo "local only"
  fi
}

current_profile_label() {
  if [[ -n "${PROFILE:-}" ]]; then
    echo "$PROFILE"
  else
    echo "none"
  fi
}

detect_default_gpu_devices() {
  local detected
  detected=$(
    list_nvidia_gpus 2>/dev/null |
      awk -F'\t' '
        BEGIN { sep = "" }
        tolower($2) ~ /2080[[:space:]]*ti/ {
          out = out sep $1
          sep = ","
          count++
          if (count == 2) {
            print out
            exit
          }
        }
      '
  ) || true
  if [[ -n "$detected" ]]; then
    printf '%s\n' "$detected"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  else
    printf '0,1\n'
  fi
}

normalize_gpu_devices_to_uuids() {
  local devices=${1:-}
  local part uuid normalized=""
  local -a parts=()

  devices=${devices// /}
  IFS=',' read -r -a parts <<< "$devices"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] || continue
    if [[ "$part" =~ ^[0-9]+$ ]]; then
      uuid=$(
        nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null |
          awk -F',' -v wanted_index="$part" '
            {
              gpu_index = $1
              gpu_uuid = $2
              gsub(/^[ \t]+|[ \t]+$/, "", gpu_index)
              gsub(/^[ \t]+|[ \t]+$/, "", gpu_uuid)
              if (gpu_index == wanted_index) {
                print gpu_uuid
                exit
              }
            }
          '
      ) || true
      [[ -n "$uuid" ]] && part="$uuid"
    fi
    if [[ -n "$normalized" ]]; then
      normalized+=","
    fi
    normalized+="$part"
  done
  printf '%s\n' "$normalized"
}

gpu_selected() {
  local idx=$1
  local devices=",${2// /},"
  [[ "$devices" == *",$idx,"* ]]
}

select_gpu_devices_menu() {
  local rows=() selected_devices idx=0 key count current_line gpu_idx gpu_name new_devices tp_count
  mapfile -t rows < <(list_nvidia_gpus || true)
  selected_devices=${GPU_DEVICES:-$(detect_default_gpu_devices)}

  if ((${#rows[@]} == 0)) || ! is_tty; then
    GPU_DEVICES=$(prompt_default "GPU devices / CUDA_VISIBLE_DEVICES" "$selected_devices") || return 0
    tp_count=$(gpu_device_count "$GPU_DEVICES")
    if (( tp_count > 0 )); then
      TP_SIZE="$tp_count"
      PP_SIZE=1
      select_parallelism_menu "$tp_count" || return 0
    fi
    save_manager_state
    return 0
  fi

  count=${#rows[@]}
  while true; do
    clear >/dev/tty
    {
      banner
      echo "GPU / TP / PP selection"
      echo
      echo "Space toggles a GPU. Enter confirms, then choose TP/PP."
      echo
      for i in "${!rows[@]}"; do
        current_line=${rows[$i]}
        gpu_idx=${current_line%%$'\t'*}
        gpu_name=${current_line#*$'\t'}
        if gpu_selected "$gpu_idx" "$selected_devices"; then
          mark="[x]"
        else
          mark="[ ]"
        fi
        if (( i == idx )); then
          printf ' > %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        else
          printf '   %s GPU %s  %s\n' "$mark" "$gpu_idx" "$gpu_name"
        fi
      done
      echo
      printf 'Selected: %s    TP_SIZE: %s    PP_SIZE: %s\n' "${selected_devices:-none}" "${TP_SIZE:-auto}" "${PP_SIZE:-1}"
    } >/dev/tty

    IFS= read -rsn1 key </dev/tty || true
    if [[ "$key" == $'\x1b' ]]; then
      read -rsn2 -t 0.1 key </dev/tty || true
      [[ -z "$key" ]] && return 0
      case "$key" in
        "[A") (( idx > 0 )) && idx=$((idx - 1)) ;;
        "[B") (( idx < count - 1 )) && idx=$((idx + 1)) ;;
      esac
    elif [[ "$key" == " " ]]; then
      current_line=${rows[$idx]}
      gpu_idx=${current_line%%$'\t'*}
      if gpu_selected "$gpu_idx" "$selected_devices"; then
        new_devices=""
        IFS=',' read -r -a parts <<< "${selected_devices// /}"
        for part in "${parts[@]}"; do
          [[ -z "$part" || "$part" == "$gpu_idx" ]] && continue
          if [[ -n "$new_devices" ]]; then
            new_devices+=",$part"
          else
            new_devices="$part"
          fi
        done
        selected_devices="$new_devices"
      else
        if [[ -n "$selected_devices" ]]; then
          selected_devices+=",$gpu_idx"
        else
          selected_devices="$gpu_idx"
        fi
      fi
    elif [[ "$key" == "" ]]; then
      if [[ -z "$selected_devices" ]]; then
        echo "Select at least one GPU." >/dev/tty
        sleep 1
        continue
      fi
      GPU_DEVICES="$selected_devices"
      tp_count=$(gpu_device_count "$GPU_DEVICES")
      TP_SIZE="$tp_count"
      PP_SIZE=1
      select_parallelism_menu "$tp_count" || return 0
      save_manager_state
      return 0
    elif [[ "$key" == "q" || "$key" == "Q" ]]; then
      return 0
    fi
  done
}

select_weight_dir() {
  local selected
  selected=$(prompt_required_dir "Weight/checkpoint directory" "${MODEL_DIR:-}") || return 0
  MODEL_DIR="$selected"
  MODEL_FAMILY=$(guess_model_family "$MODEL_DIR")
  QUANTIZATION=$(guess_quantization "$MODEL_DIR")
  SERVED_NAME=$(basename "$MODEL_DIR")
  save_manager_state
}

apply_profile_preset_menu() {
  local profiles=() selected profile_file choices=() compatible_modes
  mapfile -t profiles < <(list_profiles)
  if ((${#profiles[@]} == 0)); then
    echo "No .env profiles found under $PROFILE_DIR."
    echo
    pause_enter
    return 0
  fi
  choices=("Return" "${profiles[@]}")
  selected=$(menu_select "Profile preset" "${PROFILE:-Return}" "${choices[@]}") || return 0
  case "$selected" in
    "Return")
      return 0
      ;;
  esac
  PROFILE="$selected"
  profile_file="$PROFILE_DIR/$PROFILE"
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Applying profile preset: $PROFILE"
  echo
  profile_summary "$profile_file"
  echo
  echo "Profile applied. Use \"Edit current runtime parameters\" if you want to override fields."
  echo
  apply_profile_overrides "$profile_file"
  compatible_modes=${COMPATIBLE_MODES:-$(read_profile_value "$profile_file" COMPATIBLE_MODES)}
  normalize_mode
  if ! mode_is_compatible "${MODE:-normal}" "$compatible_modes"; then
    MODE=$(first_compatible_mode "$compatible_modes")
    echo "Launch mode switched to compatible mode: $MODE"
    echo
  fi
  save_manager_state
  pause_enter
}

save_current_profile_menu() {
  local family_dir weight_dir profile_name safe_name target_dir target_file answer compatible_modes

  if ! is_tty; then
    return 0
  fi

  family_dir=$(profile_family_dir)
  weight_dir=$(profile_weight_dir)
  target_dir="$PROFILE_DIR/$family_dir/$weight_dir/user"

  while true; do
    profile_name=$(read_line_with_esc "New profile name [${SERVED_NAME:-user-profile}] (Esc to cancel): ") || return 0
    [[ -z "$profile_name" ]] && profile_name=${SERVED_NAME:-user-profile}
    safe_name=$(sanitize_profile_name "$profile_name")
    target_file="$target_dir/${safe_name}.env"
    if [[ -e "$target_file" ]]; then
      answer=$(read_line_with_esc "Profile exists. Overwrite $target_file? [y/N]: ") || return 0
      case "$answer" in
        y|Y) break ;;
        *) continue ;;
      esac
    else
      break
    fi
  done

  mkdir -p "$target_dir"
  compatible_modes=$(profile_compatible_modes_for_current)
  : > "$target_file.tmp"
  write_profile_entry "$target_file.tmp" SERVED_NAME "${SERVED_NAME:-$safe_name}"
  write_profile_entry "$target_file.tmp" COMPATIBLE_MODES "$compatible_modes"
  write_profile_entry "$target_file.tmp" MODEL_FAMILY "${MODEL_FAMILY:-}"
  write_profile_entry "$target_file.tmp" PROFILE_GROUP "${PROFILE_GROUP:-}"
  write_profile_entry "$target_file.tmp" MODEL_VARIANT "${MODEL_VARIANT:-}"
  write_profile_entry "$target_file.tmp" QUANTIZATION "${QUANTIZATION:-}"
  write_profile_entry "$target_file.tmp" KV_CACHE_DTYPE "${KV_CACHE_DTYPE:-}"
  write_profile_entry "$target_file.tmp" MAX_MODEL_LEN "${MAX_MODEL_LEN:-}"
  write_profile_entry "$target_file.tmp" KV_CACHE_MEMORY_BYTES "${KV_CACHE_MEMORY_BYTES:-}"
  write_profile_entry "$target_file.tmp" GPU_UTIL "${GPU_UTIL:-}"
  write_profile_entry "$target_file.tmp" MAX_BATCHED_TOKENS "${MAX_BATCHED_TOKENS:-}"
  write_profile_entry "$target_file.tmp" MAX_NUM_SEQS "${MAX_NUM_SEQS:-}"
  write_profile_entry "$target_file.tmp" LONG_PREFILL_TOKEN_THRESHOLD "${LONG_PREFILL_TOKEN_THRESHOLD:-}"
  write_profile_entry "$target_file.tmp" PREFILL_BATCH_BARRIER "${PREFILL_BATCH_BARRIER:-0}"
  write_profile_entry "$target_file.tmp" DISABLE_PREFIX_CACHING "${DISABLE_PREFIX_CACHING:-0}"
  write_profile_entry "$target_file.tmp" MTP_K "${MTP_K:-}"
  write_profile_entry "$target_file.tmp" VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-}"
  write_profile_entry "$target_file.tmp" VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS "${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-}"
  write_profile_entry "$target_file.tmp" VLLM_TURBOQUANT_MAX_KV_SPLITS "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-}"
  write_profile_entry "$target_file.tmp" VLLM_TURBOQUANT_DECODE_BLOCK_KV "${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-}"
  write_profile_entry "$target_file.tmp" MESSAGE_TYPE "${MESSAGE_TYPE:-}"
  write_profile_entry "$target_file.tmp" MM_LIMIT_JSON "${MM_LIMIT_JSON:-}"
  write_profile_entry "$target_file.tmp" LANGUAGE_MODEL_ONLY "${LANGUAGE_MODEL_ONLY:-}"
  write_profile_entry "$target_file.tmp" SKIP_MM_PROFILING "${SKIP_MM_PROFILING:-}"
  write_profile_entry "$target_file.tmp" HF_OVERRIDES_JSON "${HF_OVERRIDES_JSON:-}"
  write_profile_entry "$target_file.tmp" ADDITIONAL_CONFIG_JSON "${ADDITIONAL_CONFIG_JSON:-}"
  write_profile_entry "$target_file.tmp" SPECULATIVE_CONFIG "${SPECULATIVE_CONFIG:-}"
  write_profile_entry "$target_file.tmp" ATTENTION_BACKEND "${ATTENTION_BACKEND:-}"
  write_profile_entry "$target_file.tmp" DISABLE_HYBRID_KV_CACHE_MANAGER "${DISABLE_HYBRID_KV_CACHE_MANAGER:-}"
  write_profile_entry "$target_file.tmp" DISABLE_CUSTOM_ALL_REDUCE "${DISABLE_CUSTOM_ALL_REDUCE:-}"
  mv "$target_file.tmp" "$target_file"

  PROFILE="$family_dir/$weight_dir/user/${safe_name}.env"
  save_manager_state
  echo "Saved profile: $target_file"
  echo "Compatible mode: $compatible_modes"
  echo
  pause_enter
}

change_profile_dir_menu() {
  local selected_dir
  selected_dir=$(prompt_required_dir "Profile directory" "${PROFILE_DIR:-$MANAGER_ROOT/profiles}") || return 0
  PROFILE_DIR="$selected_dir"
  TEMPLATE_DIR="$PROFILE_DIR/templates"
  save_manager_state
}

change_template_dir_menu() {
  local selected_dir
  selected_dir=$(prompt_required_dir "Template directory" "${TEMPLATE_DIR:-$PROFILE_DIR/templates}") || return 0
  TEMPLATE_DIR="$selected_dir"
  save_manager_state
}

select_template_preset_menu() {
  local templates=() choices=() selected resolved
  mapfile -t templates < <(list_template_presets)
  choices=("model default")
  if ((${#templates[@]} > 0)); then
    choices+=("${templates[@]}")
  fi
  choices+=("manual path" "change template directory" "Return")

  selected=$(menu_select "Chat template preset" "$(current_template_label)" "${choices[@]}") || return 0
  case "$selected" in
    "model default")
      CHAT_TEMPLATE_PRESET=""
      CHAT_TEMPLATE_FILE=""
      save_manager_state
      ;;
    "manual path")
      CHAT_TEMPLATE_FILE=$(prompt_optional "Chat template file" "${CHAT_TEMPLATE_FILE:-}") || return 0
      CHAT_TEMPLATE_PRESET=""
      save_manager_state
      ;;
    "change template directory")
      change_template_dir_menu
      ;;
    "Return")
      return 0
      ;;
    *)
      if resolved=$(resolve_template_file "$selected"); then
        CHAT_TEMPLATE_PRESET="$selected"
        CHAT_TEMPLATE_FILE="$resolved"
        save_manager_state
      else
        echo "Template preset not found: $selected"
        echo
        pause_enter
      fi
      ;;
  esac
}

edit_reasoning_defaults_menu() {
  local selected choices=()
  choices=(
    "template default"
    "thinking on"
    "thinking off"
    "custom kwargs"
    "reasoning parser"
    "default thinking budget"
    "Return"
  )
  selected=$(menu_select "Reasoning defaults" "$(current_reasoning_label)" "${choices[@]}") || return 0
  case "$selected" in
    "template default")
      REASONING_MODE=""
      DEFAULT_CHAT_TEMPLATE_KWARGS=""
      save_manager_state
      ;;
    "thinking on")
      REASONING_MODE=on
      DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
      save_manager_state
      ;;
    "thinking off")
      REASONING_MODE=off
      DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
      save_manager_state
      ;;
    "custom kwargs")
      DEFAULT_CHAT_TEMPLATE_KWARGS=$(prompt_optional "Default chat template kwargs JSON" "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}") || return 0
      REASONING_MODE=custom
      save_manager_state
      ;;
    "reasoning parser")
      echo "Use off to disable the automatic parser default for diagnostics."
      REASONING_PARSER=$(prompt_optional "Reasoning parser" "${REASONING_PARSER:-}") || return 0
      save_manager_state
      ;;
    "default thinking budget")
      REASONING_BUDGET=$(prompt_optional "Default thinking token budget" "${REASONING_BUDGET:-}") || return 0
      echo "Default budget is applied only when a chat request does not provide thinking_token_budget."
      echo
      save_manager_state
      pause_enter
      ;;
    "Return")
      return 0
      ;;
  esac
}

edit_tool_calling_menu() {
  local selected choices=()
  choices=(
    "off"
    "auto tool choice"
    "tool call parser"
    "tool parser plugin"
    "Return"
  )
  selected=$(menu_select "Tool calling defaults" "$(current_tool_calling_label)" "${choices[@]}") || return 0
  case "$selected" in
    "off")
      ENABLE_AUTO_TOOL_CHOICE=0
      save_manager_state
      ;;
    "auto tool choice")
      ENABLE_AUTO_TOOL_CHOICE=1
      if [[ -z "${TOOL_CALL_PARSER:-}" ]]; then
        TOOL_CALL_PARSER=$(prompt_default "Tool call parser" "qwen3_xml") || return 0
      fi
      VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}
      save_manager_state
      ;;
    "tool call parser")
      TOOL_CALL_PARSER=$(prompt_optional "Tool call parser" "${TOOL_CALL_PARSER:-}") || return 0
      save_manager_state
      ;;
    "tool parser plugin")
      TOOL_PARSER_PLUGIN=$(prompt_optional "Tool parser plugin module path" "${TOOL_PARSER_PLUGIN:-}") || return 0
      save_manager_state
      ;;
    "Return")
      return 0
      ;;
  esac
}

select_profile_preset() {
  local selected choices=()

  while true; do
    if is_tty; then
      clear >/dev/tty 2>/dev/null || true
    fi
    banner
    echo "Profile manager"
    echo
    echo "Profiles are presets. Applying one fills the runtime parameters, then you"
    echo "can override every field before launching."
    echo
    printf '  Current profile dir: %s\n' "${PROFILE_DIR:-$MANAGER_ROOT/profiles}"
    printf '  Current profile:     %s\n' "$(current_profile_label)"
    printf '  Template dir:        %s\n' "${TEMPLATE_DIR:-$PROFILE_DIR/templates}"
    printf '  Chat template:       %s\n' "$(current_template_label)"
    printf '  Reasoning default:   %s\n' "$(current_reasoning_label)"
    printf '  Tool calling:        %s\n' "$(current_tool_calling_label)"
    echo

    choices=(
      "Apply profile preset"
      "Chat template preset"
      "Reasoning defaults"
      "Tool calling defaults"
      "Edit current runtime parameters"
      "Save current profile"
      "Change profile directory"
      "Clear profile"
      "Show profile list"
      "Return"
    )
    selected=$(menu_select "Profile action" "Apply profile preset" "${choices[@]}") || return 0
    case "$selected" in
      "Apply profile preset")
        apply_profile_preset_menu
        ;;
      "Chat template preset")
        select_template_preset_menu
        ;;
      "Reasoning defaults")
        edit_reasoning_defaults_menu
        ;;
      "Tool calling defaults")
        edit_tool_calling_menu
        ;;
      "Edit current runtime parameters")
        runtime_parameter_menu
        ;;
      "Save current profile")
        save_current_profile_menu
        ;;
      "Change profile directory")
        change_profile_dir_menu
        ;;
      "Clear profile")
        PROFILE=""
        save_manager_state
        ;;
      "Show profile list")
        show_profiles
        ;;
      "Return")
        return 0
        ;;
    esac
  done
}

prompt_optional() {
  local label=$1
  local default=${2:-}
  local answer

  if ! is_tty; then
    printf '%s\n' "$default"
    return 0
  fi

  answer=$(read_line_with_esc "$label [${default:-empty}] (type '-' to clear, Esc to cancel): ") || return 130
  if [[ "$answer" == "-" ]]; then
    printf '\n'
  elif [[ -z "$answer" ]]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$answer"
  fi
}

prompt_toggle01() {
  local label=$1
  local default=${2:-0}
  local answer

  while true; do
    answer=$(read_line_with_esc "$label [$default] (0/1, Enter to keep, Esc to cancel): ") || return 130
    case "$answer" in
      "")
        printf '%s\n' "$default"
        return 0
        ;;
      0|1)
        printf '%s\n' "$answer"
        return 0
        ;;
      *)
        echo "Please enter 0 or 1." >&2
        ;;
    esac
  done
}

edit_advanced_parameters() {
  local answer

  if ! is_tty; then
    return 0
  fi

  answer=$(read_line_with_esc "Edit advanced optional parameters? [y/N]: ") || return 0
  case "$answer" in
    y|Y) ;;
    *) return 0 ;;
  esac

  echo
  ATTENTION_BACKEND=$(prompt_optional "Attention backend" "${ATTENTION_BACKEND:-}") || return 0
  HF_OVERRIDES_JSON=$(prompt_optional "HF overrides JSON" "${HF_OVERRIDES_JSON:-}") || return 0
  ADDITIONAL_CONFIG_JSON=$(prompt_optional "Additional config JSON" "${ADDITIONAL_CONFIG_JSON:-}") || return 0
  SPECULATIVE_CONFIG=$(prompt_optional "Speculative config JSON" "${SPECULATIVE_CONFIG:-}") || return 0
  COMPILATION_CONFIG_JSON=$(prompt_optional "Compilation config JSON" "${COMPILATION_CONFIG_JSON:-}") || return 0
  MM_LIMIT_JSON=$(prompt_optional "Multimodal limit JSON" "${MM_LIMIT_JSON:-}") || return 0
  if [[ -n "${MM_LIMIT_JSON:-}" && "${MESSAGE_TYPE:-text-only}" == "text-only" ]]; then
    MESSAGE_TYPE=text+image
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=0
  fi
  LANGUAGE_MODEL_ONLY=$(prompt_toggle01 "Language-model only" "${LANGUAGE_MODEL_ONLY:-1}") || return 0
  SKIP_MM_PROFILING=$(prompt_toggle01 "Skip multimodal profiling" "${SKIP_MM_PROFILING:-1}") || return 0
  ENFORCE_EAGER=$(prompt_toggle01 "Enforce eager" "${ENFORCE_EAGER:-0}") || return 0
  CONFIG_OVERRIDE_SOURCE["ENFORCE_EAGER"]=menu
  unset 'CONFIG_OVERRIDE_UNSET[ENFORCE_EAGER]'
  NO_ASYNC_SCHEDULING=$(prompt_toggle01 "No async scheduling" "${NO_ASYNC_SCHEDULING:-0}") || return 0
  DISABLE_HYBRID_KV_CACHE_MANAGER=$(prompt_toggle01 "Disable hybrid KV cache manager" "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}") || return 0
  DISABLE_PREFIX_CACHING=$(prompt_toggle01 "Disable prefix caching" "${DISABLE_PREFIX_CACHING:-0}") || return 0
  DISABLE_CUSTOM_ALL_REDUCE=$(prompt_toggle01 "Disable custom all-reduce" "${DISABLE_CUSTOM_ALL_REDUCE:-0}") || return 0
  DISABLE_LOG_STATS=$(prompt_toggle01 "Disable log stats" "${DISABLE_LOG_STATS:-0}") || return 0
  CONFIG_OVERRIDE_SOURCE["DISABLE_LOG_STATS"]=menu
  unset 'CONFIG_OVERRIDE_UNSET[DISABLE_LOG_STATS]'
  normalize_message_type_defaults
}

edit_runtime_parameters() {
  local answer kv_choice message_choice current_message_type

  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
    banner
    echo "Runtime parameters"
    echo
    echo "Press Enter to keep the current value. Type '-' on optional fields to clear."
    echo
  fi

  MODEL_FAMILY=$(prompt_default "Model family" "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}") || return 0
  PROFILE_GROUP=$(prompt_optional "Profile group" "${PROFILE_GROUP:-}") || return 0
  MODEL_VARIANT=$(prompt_optional "Weight precision/profile variant" "${MODEL_VARIANT:-}") || return 0
  SERVED_NAME=$(prompt_default "Served model name" "${SERVED_NAME:-${MODEL_DIR:+$(basename "$MODEL_DIR")}}") || return 0
  QUANTIZATION=$(prompt_default "vLLM --quantization (empty/auto, fp8, gptq_marlin, awq_marlin, compressed-tensors, quark)" "${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}") || return 0

  kv_choice=$(menu_select "KV precision" "${KV_CACHE_DTYPE:-fp16}" fp16 int8_per_token_head turboquant_k8v4 turboquant_4bit_nc) || return 0
  if [[ "$kv_choice" == "fp16" ]]; then
    KV_CACHE_DTYPE=""
  else
    KV_CACHE_DTYPE="$kv_choice"
  fi

  MAX_MODEL_LEN=$(prompt_default "Context tokens" "${MAX_MODEL_LEN:-$(default_context_tokens)}") || return 0
  KV_CACHE_MEMORY_BYTES=$(prompt_optional "KV cache memory bytes (empty = use GPU util)" "${KV_CACHE_MEMORY_BYTES:-}") || return 0
  GPU_UTIL=$(prompt_default "GPU memory utilization" "${GPU_UTIL:-$(default_gpu_util)}") || return 0
  MAX_BATCHED_TOKENS=$(prompt_default "Max batched tokens" "${MAX_BATCHED_TOKENS:-2048}") || return 0
  MAX_NUM_SEQS=$(prompt_default "Max concurrent sequences" "${MAX_NUM_SEQS:-1}") || return 0
  MTP_K=$(prompt_default "MTP speculative tokens" "${MTP_K:-0}") || return 0

  current_message_type=${MESSAGE_TYPE:-text-only}
  [[ -n "${MM_LIMIT_JSON:-}" ]] && current_message_type=text+image
  message_choice=$(menu_select "Message type" "$current_message_type" "text-only" "text+image") || return 0
  MESSAGE_TYPE="$message_choice"
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=0
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  edit_advanced_parameters
  normalize_message_type_defaults

  save_manager_state
}

edit_kv_precision_menu() {
  local kv_choice
  kv_choice=$(menu_select "KV precision" "${KV_CACHE_DTYPE:-fp16}" fp16 int8_per_token_head turboquant_k8v4 turboquant_4bit_nc) || return 0
  if [[ "$kv_choice" == "fp16" ]]; then
    KV_CACHE_DTYPE=""
  else
    KV_CACHE_DTYPE="$kv_choice"
  fi
  save_manager_state
}

edit_message_type_menu() {
  local current_message_type message_choice
  current_message_type=${MESSAGE_TYPE:-text-only}
  [[ -n "${MM_LIMIT_JSON:-}" ]] && current_message_type=text+image
  message_choice=$(menu_select "Message type" "$current_message_type" "text-only" "text+image") || return 0
  MESSAGE_TYPE="$message_choice"
  if [[ "$MESSAGE_TYPE" == "text+image" ]]; then
    MM_LIMIT_JSON=${MM_LIMIT_JSON:-'{"image":1,"video":0,"audio":0}'}
    LANGUAGE_MODEL_ONLY=0
    SKIP_MM_PROFILING=0
  else
    MM_LIMIT_JSON=""
    LANGUAGE_MODEL_ONLY=1
    SKIP_MM_PROFILING=1
  fi
  normalize_message_type_defaults
  save_manager_state
}

edit_prefix_cache_menu() {
  local choice

  choice=$(menu_select "Prefix cache" "$(current_prefix_cache_label)" enabled disabled) || return 0
  case "$choice" in
    disabled)
      DISABLE_PREFIX_CACHING=1
      ENABLE_PREFIX_CACHING=0
      ;;
    *)
      ENABLE_PREFIX_CACHING=1
      DISABLE_PREFIX_CACHING=0
      ENABLE_PROMPT_TOKENS_DETAILS=1
      ;;
  esac
  save_manager_state
}

runtime_parameter_menu() {
  local selected choices=()
  local model_family_value profile_group_value model_variant_value served_name_value
  local quantization_value kv_value context_value kv_cache_memory_value gpu_util_value
  local batch_tokens_value max_sequences_value mtp_value message_type_value
  local template_value reasoning_value tool_calling_value prefix_cache_value

  while true; do
    model_family_value=$(menu_value "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}")
    profile_group_value=$(menu_value "${PROFILE_GROUP:-}")
    model_variant_value=$(menu_value "${MODEL_VARIANT:-}")
    served_name_value=$(menu_value "${SERVED_NAME:-}")
    quantization_value=$(menu_value "${QUANTIZATION:-auto}")
    kv_value=$(menu_value "${KV_CACHE_DTYPE:-fp16}")
    context_value=$(menu_value "${MAX_MODEL_LEN:-$(default_context_tokens)}")
    kv_cache_memory_value=$(menu_value "${KV_CACHE_MEMORY_BYTES:-auto}")
    gpu_util_value=$(menu_value "${GPU_UTIL:-$(default_gpu_util)}")
    if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
      gpu_util_value="${gpu_util_value} (fallback)"
    fi
    batch_tokens_value=$(menu_value "${MAX_BATCHED_TOKENS:-2048}")
    max_sequences_value=$(menu_value "${MAX_NUM_SEQS:-1}")
    mtp_value=$(menu_value "${MTP_K:-0}")
    message_type_value=$(menu_value "${MESSAGE_TYPE:-text-only}")
    template_value=$(menu_value "$(current_template_label)")
    reasoning_value=$(menu_value "$(current_reasoning_label)")
    tool_calling_value=$(menu_value "$(current_tool_calling_label)")
    prefix_cache_value=$(menu_value "$(current_prefix_cache_label)")

    if is_tty; then
      clear >/dev/tty 2>/dev/null || true
    fi
    banner
    echo "Runtime parameter overrides"
    echo
    choices=(
      "Model family: $model_family_value"
      "Profile group: $profile_group_value"
      "Weight variant: $model_variant_value"
      "Served name: $served_name_value"
      "vLLM --quantization: $quantization_value"
      "KV precision: $kv_value"
      "Context tokens: $context_value"
      "KV cache memory bytes: $kv_cache_memory_value"
      "GPU util: $gpu_util_value"
      "Batch tokens: $batch_tokens_value"
      "Max sequences: $max_sequences_value"
      "MTP tokens: $mtp_value"
      "Message type: $message_type_value"
      "Chat template: $template_value"
      "Reasoning defaults: $reasoning_value"
      "Tool calling: $tool_calling_value"
      "Prefix cache: $prefix_cache_value"
      "Advanced options"
      "Edit all fields"
      "Return"
    )
    selected=$(menu_select "Runtime parameter" "Return" "${choices[@]}") || return 0
    case "$selected" in
      "Model family:"*)
        MODEL_FAMILY=$(prompt_default "Model family" "${MODEL_FAMILY:-$(guess_model_family "${MODEL_DIR:-}")}") || continue
        save_manager_state
        ;;
      "Profile group:"*)
        PROFILE_GROUP=$(prompt_optional "Profile group" "${PROFILE_GROUP:-}") || continue
        save_manager_state
        ;;
      "Weight variant:"*)
        MODEL_VARIANT=$(prompt_optional "Weight precision/profile variant" "${MODEL_VARIANT:-}") || continue
        save_manager_state
        ;;
      "Served name:"*)
        SERVED_NAME=$(prompt_default "Served model name" "${SERVED_NAME:-${MODEL_DIR:+$(basename "$MODEL_DIR")}}") || continue
        save_manager_state
        ;;
      "vLLM --quantization:"*)
        QUANTIZATION=$(prompt_default "vLLM --quantization (empty/auto, fp8, gptq_marlin, awq_marlin, compressed-tensors, quark)" "${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}") || continue
        save_manager_state
        ;;
      "KV precision:"*)
        edit_kv_precision_menu
        ;;
      "Context tokens:"*)
        MAX_MODEL_LEN=$(prompt_default "Context tokens" "${MAX_MODEL_LEN:-$(default_context_tokens)}") || continue
        save_manager_state
        ;;
      "KV cache memory bytes:"*)
        KV_CACHE_MEMORY_BYTES=$(prompt_optional "KV cache memory bytes (empty = use GPU util)" "${KV_CACHE_MEMORY_BYTES:-}") || continue
        save_manager_state
        ;;
      "GPU util:"*)
        GPU_UTIL=$(prompt_default "GPU memory utilization" "${GPU_UTIL:-$(default_gpu_util)}") || continue
        save_manager_state
        ;;
      "Batch tokens:"*)
        MAX_BATCHED_TOKENS=$(prompt_default "Max batched tokens" "${MAX_BATCHED_TOKENS:-2048}") || continue
        save_manager_state
        ;;
      "Max sequences:"*)
        MAX_NUM_SEQS=$(prompt_default "Max concurrent sequences" "${MAX_NUM_SEQS:-1}") || continue
        save_manager_state
        ;;
      "MTP tokens:"*)
        MTP_K=$(prompt_default "MTP speculative tokens" "${MTP_K:-0}") || continue
        save_manager_state
        ;;
      "Message type:"*)
        edit_message_type_menu
        ;;
      "Chat template:"*)
        select_template_preset_menu
        ;;
      "Reasoning defaults:"*)
        edit_reasoning_defaults_menu
        ;;
      "Tool calling:"*)
        edit_tool_calling_menu
        ;;
      "Prefix cache:"*)
        edit_prefix_cache_menu
        ;;
      "Advanced options")
        edit_advanced_parameters
        save_manager_state
        ;;
      "Edit all fields")
        edit_runtime_parameters
        ;;
      "Return")
        return 0
        ;;
    esac
  done
}

normalize_mode() {
  # Compatibility shim for older state files or scripts.
  case "${MODE:-}" in
    stable)
      MODE=safe
      ;;
    speed)
      MODE=normal
      ;;
  esac
}

select_mode_menu() {
  normalize_mode
  MODE=$(prompt_segmented "Launch mode" "${MODE:-normal}" safe normal fast aggressive) || return 0
  save_manager_state
}

input_port_menu() {
  local current answer
  current=${PORT:-8000}

  if ! is_tty; then
    PORT="$current"
    save_manager_state
    return 0
  fi

  while true; do
    answer=$(read_line_with_esc "Port [$current] (q to cancel, Esc to return): ") || return 0
    case "${answer,,}" in
      q|quit|exit)
        return 0
        ;;
    esac
    [[ -z "$answer" ]] && answer="$current"
    if [[ "$answer" =~ ^[0-9]+$ ]] && (( answer >= 1 && answer <= 65535 )); then
      PORT="$answer"
      break
    fi
    echo "Please enter a port from 1 to 65535." >&2
  done
  save_manager_state
}

select_scope_menu() {
  local selected
  selected=$(prompt_segmented "Service scope" "$(current_scope_label)" "local only" "local + LAN") || return 0
  if [[ "$selected" == "local + LAN" ]]; then
    SERVICE_SCOPE=lan
  else
    SERVICE_SCOPE=local
  fi
  save_manager_state
}

show_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Service status:"
  echo
  mkdir -p "$LOG_DIR"
  local found=0 pid_file pid name state cmd
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    found=1
    name=$(basename "$pid_file" .pid)
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      state="running"
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | sed 's/[[:space:]]*$//')
    else
      state="stale"
      cmd="-"
    fi
    printf '  %-36s pid=%-8s %s\n' "$name" "${pid:-unknown}" "$state"
    [[ "$cmd" != "-" ]] && printf '    %s\n' "$cmd"
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  if (( found == 0 )); then
    echo "  No pid files found under $LOG_DIR."
  fi
  echo
  pause_enter
}

show_launch_status() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  local info pid_file pid name
  echo "Service status"
  echo
  echo "  Status:       START OK"
  echo "  Served model: ${SERVED_NAME:-unknown}"
  echo "  Model path:   ${MODEL_DIR:-unknown}"
  echo "  GPU devices:  ${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-unknown}}"
  echo "  Mode:         ${MODE:-safe}"
  echo "  Scope:        ${SERVICE_SCOPE:-local}"
  echo "  Local API:    ${LAST_API_LOCAL:-http://127.0.0.1:${PORT:-8000}/v1}"
  if [[ -n "${LAST_API_LAN:-}" ]]; then
    echo "  LAN API:      $LAST_API_LAN"
  fi
  echo "  PID file:     ${LAST_PID_FILE:-unknown}"
  echo "  Log file:     ${LAST_LOG_FILE:-unknown}"
  if info=$(current_service_info); then
    IFS=$'\t' read -r pid_file pid name <<< "$info"
    render_kv_cache_status "$pid_file" "$pid" "$name" 13
  fi
  if [[ -n "${LAST_SMOKE_OUTPUT:-}" ]]; then
    echo "  Smoke:        $LAST_SMOKE_OUTPUT"
  fi
}

print_running_services() {
  local pid_file pid name
  [[ -d "$LOG_DIR" ]] || return 0
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    pid_is_running "$pid" || continue
    name=$(pid_file_service_name "$pid_file")
    printf '  %s pid=%s\n' "$name" "$pid"
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
}

clear_last_service_state() {
  LAST_PID_FILE=""
  LAST_LOG_FILE=""
  LAST_API_LOCAL=""
  LAST_API_LAN=""
  LAST_SMOKE_OUTPUT=""
  save_manager_state
}

stop_pid_file() {
  local pid_file=$1
  local pid descendants
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if ! pid_is_running "$pid"; then
    rm -f "$pid_file"
    return 0
  fi

  descendants=$(collect_descendant_pids "$pid")

  stop_pid_tree "$pid" || true
  for _ in {1..20}; do
    pid_is_running "$pid" || {
      stop_recorded_pids "$descendants" || true
      wait_recorded_pids_stopped "$descendants" 10 || true
      stop_recorded_pids "$descendants" force || true
      wait_recorded_pids_stopped "$descendants" 10 || true
      cleanup_vllm_worker_residuals || true
      rm -f "$pid_file"
      return 0
    }
    sleep 0.5
  done

  stop_pid_tree "$pid" force || true
  for _ in {1..20}; do
    pid_is_running "$pid" || {
      stop_recorded_pids "$descendants" force || true
      wait_recorded_pids_stopped "$descendants" 10 || true
      cleanup_vllm_worker_residuals force || true
      rm -f "$pid_file"
      return 0
    }
    sleep 0.5
  done

  return 1
}

collect_descendant_pids() {
  local pid=${1:-}
  local child

  pid_is_running "$pid" || return 0
  while read -r child; do
    [[ -n "$child" ]] || continue
    printf '%s\n' "$child"
    collect_descendant_pids "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
}

stop_recorded_pids() {
  local pids=${1:-}
  local mode=${2:-term}
  local signal pid

  signal=TERM
  [[ "$mode" == "force" ]] && signal=KILL
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    pid_is_running "$pid" || continue
    kill "-$signal" "$pid" 2>/dev/null || true
  done <<<"$pids"
}

wait_recorded_pids_stopped() {
  local pids=${1:-}
  local attempts=${2:-20}
  local pid any

  for _ in $(seq 1 "$attempts"); do
    any=0
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      pid_is_running "$pid" || continue
      any=1
      break
    done <<<"$pids"
    (( any == 0 )) && return 0
    sleep 0.5
  done
  return 1
}

stop_pid_tree() {
  local pid=${1:-}
  local mode=${2:-term}
  local pgid signal children child

  pid_is_running "$pid" || return 0
  signal=TERM
  [[ "$mode" == "force" ]] && signal=KILL
  children=$(pgrep -P "$pid" 2>/dev/null || true)
  for child in $children; do
    stop_pid_tree "$child" "$mode" || true
  done

  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill "-$signal" -- "-$pgid" 2>/dev/null || true
  else
    kill "-$signal" "$pid" 2>/dev/null || true
  fi
}

cleanup_vllm_residuals() {
  local port=${1:-}
  local served_name=${2:-}
  local model_dir=${3:-}
  local pid cmd

  while IFS= read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] || continue
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]] || continue
    if [[ -n "$port" && "$cmd" != *"--port $port"* && "$cmd" != *"--port=${port}"* ]]; then
      if [[ -z "$served_name" || "$cmd" != *"$served_name"* ]]; then
        if [[ -z "$model_dir" || "$cmd" != *"$model_dir"* ]]; then
          continue
        fi
      fi
    fi
    stop_pid_tree "$pid" || true
  done < <(pgrep -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true)
}

cleanup_vllm_worker_residuals() {
  local mode=${1:-term}
  local signal owner pid user comm

  signal=TERM
  [[ "$mode" == "force" ]] && signal=KILL
  owner=$(id -un)
  while read -r pid user comm; do
    [[ -n "$pid" && "$pid" != "$$" ]] || continue
    [[ "$user" == "$owner" ]] || continue
    case "$comm" in
      VLLM::*|python)
        ;;
      *)
        continue
        ;;
    esac
    if [[ "$comm" != VLLM::* ]]; then
      tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q 'multiprocessing.resource_tracker' || continue
    fi
    kill "-$signal" "$pid" 2>/dev/null || true
  done < <(ps -eo pid=,user=,comm= 2>/dev/null)
}

cleanup_failed_launch() {
  local pid_file=${1:-}
  local pid

  if [[ -n "$pid_file" && -f "$pid_file" ]]; then
    pid=$(cat "$pid_file" 2>/dev/null || true)
    stop_pid_tree "$pid" || true
    sleep 1
    stop_pid_tree "$pid" force || true
    rm -f "$pid_file"
  fi
  cleanup_vllm_residuals "${PORT:-}" "${SERVED_NAME:-}" "${MODEL_DIR:-}" || true
  cleanup_vllm_worker_residuals || true
  sleep 1
  cleanup_vllm_worker_residuals force || true
}

stop_all_managed_services() {
  local pid_file failed=0
  [[ -d "$LOG_DIR" ]] || return 0
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    stop_pid_file "$pid_file" || failed=1
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  return "$failed"
}

confirm_restart_existing() {
  local answer

  current_service_info >/dev/null || return 0
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "A managed vLLM service is already running:"
  print_running_services
  echo
  answer=$(read_line_with_esc "Restart it with the current profile/config? [y/N]: ") || return 1
  case "$answer" in
    y|Y)
      echo "Stopping existing managed service..."
      stop_all_managed_services || {
        echo "Existing service did not stop cleanly. Use item 9 to inspect/stop it."
        return 1
      }
      clear_last_service_state
      echo "Existing service stopped."
      return 0
      ;;
    *)
      echo "Restart cancelled."
      return 1
      ;;
  esac
}

show_logs() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Recent logs:"
  echo
  mkdir -p "$LOG_DIR"
  local logs=() selected log_file
  mapfile -t logs < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -printf '%T@ %f\n' 2>/dev/null | sort -nr | awk '{print $2}' | head -n 20)
  if ((${#logs[@]} == 0)); then
    echo "No logs found under $LOG_DIR."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Log file" "${logs[0]}" "${logs[@]}") || return 0
  log_file="$LOG_DIR/$selected"
  clear >/dev/tty 2>/dev/null || true
  banner
  echo "Log: $log_file"
  echo
  tail -n "${LOG_TAIL_LINES:-120}" "$log_file" || true
  echo
  pause_enter
}

stop_service() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  mkdir -p "$LOG_DIR"
  local pid_files=() target_pids=() choices=() seen_pids=() pid_file selected pid answer cmd port served_name
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      pid_files+=("$pid_file")
      target_pids+=("$pid")
      seen_pids+=("$pid")
      choices+=("$(basename "$pid_file" .pid) pid=$pid")
    fi
  done < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)
  while IFS= read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] || continue
    if ((${#seen_pids[@]} > 0)) && printf '%s\n' "${seen_pids[@]}" | grep -qx "$pid"; then
      continue
    fi
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]] || continue
    port=$(pid_arg_value "$pid" --port 2>/dev/null || true)
    served_name=$(pid_arg_value "$pid" --served-model-name 2>/dev/null || true)
    pid_files+=("")
    target_pids+=("$pid")
    choices+=("orphan-vllm ${served_name:-unknown} port=${port:-unknown} pid=$pid")
  done < <(pgrep -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true)
  if ((${#choices[@]} == 0)); then
    while read -r pid user comm; do
      [[ -n "$pid" && "$pid" != "$$" ]] || continue
      [[ "$user" == "$(id -un)" ]] || continue
      if [[ "$comm" != VLLM::* ]]; then
        tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q 'multiprocessing.resource_tracker' || continue
      fi
      pid_files+=("")
      target_pids+=("$pid")
      choices+=("orphan-vllm-worker ${comm:-unknown} pid=$pid")
    done < <(ps -eo pid=,user=,comm= 2>/dev/null)
  fi
  if ((${#choices[@]} == 0)); then
    echo "No running services found from pid files or vLLM process scan."
    echo
    pause_enter
    return 0
  fi
  selected=$(menu_select "Stop service" "${choices[0]}" "${choices[@]}") || return 0
  local index=-1
  for i in "${!choices[@]}"; do
    if [[ "${choices[$i]}" == "$selected" ]]; then
      index=$i
      break
    fi
  done
  (( index >= 0 )) || return 0
  pid_file="${pid_files[$index]}"
  pid="${target_pids[$index]}"
  if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
    echo "Service is no longer running."
    [[ -n "$pid_file" ]] && rm -f "$pid_file"
    pause_enter
    return 0
  fi
  answer=$(read_line_with_esc "Stop $selected? [y/N]: ") || {
    echo "Stop cancelled."
    echo
    pause_enter
    return 0
  }
  case "$answer" in
    y|Y)
      if [[ -n "$pid_file" ]]; then
        stop_pid_file "$pid_file"
      else
        stop_pid_tree "$pid" || true
        sleep 1
        stop_pid_tree "$pid" force || true
      fi
      cleanup_vllm_worker_residuals || true
      sleep 1
      cleanup_vllm_worker_residuals force || true
      if ! pid_is_running "$pid"; then
        if [[ "${LAST_PID_FILE:-}" == "$pid_file" ]]; then
          clear_last_service_state
        fi
        echo "Stopped."
      else
        echo "Stop requested, but the process is still running."
      fi
      ;;
    *)
      echo "Stop cancelled."
      ;;
  esac
  echo
  pause_enter
}

guess_model_family() {
  local dir=${1,,}
  if [[ "$dir" == *gemma* ]]; then
    echo gemma4
  else
    echo qwen
  fi
}

guess_quantization() {
  local dir=${1,,}
  if [[ "$dir" == *fp8* ]]; then
    echo fp8
  elif [[ "$dir" == *gptq* ]]; then
    echo gptq_marlin
  elif [[ "$dir" == *awq* ]]; then
    echo awq_marlin
  elif [[ "$dir" == *quark* ]]; then
    echo quark
  else
    echo ""
  fi
}

guess_precision_scheme() {
  local dir=${1:-${MODEL_DIR:-}}
  local quantization=${2:-${QUANTIZATION:-$(guess_quantization "$dir")}}
  dir=${dir,,}

  if [[ "$dir" == *w8a8* || "$quantization" == "quark" ]]; then
    echo W8A8
  elif [[ "$dir" == *nvfp4* || "$dir" == *mxfp4* ]]; then
    echo W4A16
  elif [[ "$dir" == *fp8* || "$quantization" == "fp8" ]]; then
    echo W8A16
  elif [[ "$dir" == *w8a16* || "$dir" == *int8* ]]; then
    echo W8A16
  elif [[ "$dir" == *w4a16* || "$dir" == *int4* || "$dir" == *4bit* || "$dir" == *awq* ]]; then
    echo W4A16
  elif [[ "$quantization" == "awq_marlin" || "$quantization" == "gptq_marlin" ]]; then
    echo W4A16
  else
    echo auto
  fi
}

default_context_tokens() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 102400
  elif [[ "$quantization" == "quark" ]]; then
    echo 8192
  else
    echo 131072
  fi
}

default_gpu_util() {
  local quantization=${QUANTIZATION:-$(guess_quantization "${MODEL_DIR:-}")}
  if [[ "$quantization" == "fp8" ]]; then
    echo 0.92
  else
    echo 0.90
  fi
}

set_derived_default() {
  local key=$1
  local value=$2
  config_key_has_explicit_value "$key" && return 0
  if [[ -n "${!key+x}" && -n "${!key}" ]]; then
    return 0
  fi
  printf -v "$key" '%s' "$value"
  export "$key"
}

set_mode_default() {
  local key=$1
  local value=$2
  config_key_has_explicit_value "$key" && return 0
  printf -v "$key" '%s' "$value"
  export "$key"
  CONFIG_OVERRIDE_SOURCE["$key"]=mode
  unset "CONFIG_OVERRIDE_UNSET[$key]"
}

apply_mode() {
  normalize_mode
  case "$MODE" in
    normal)
      set_mode_default ENFORCE_EAGER 0
      set_mode_default DISABLE_LOG_STATS 1
      set_mode_default VLLM_SM75_SPEC_SYNC_MODE safe
      set_mode_default VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH 0
      ;;
    fast)
      set_mode_default ENFORCE_EAGER 0
      set_mode_default DISABLE_LOG_STATS 1
      set_mode_default VLLM_SM75_SPEC_SYNC_MODE safe
      set_mode_default VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH 1
      ;;
    aggressive)
      set_mode_default ENFORCE_EAGER 0
      set_mode_default DISABLE_LOG_STATS 1
      set_mode_default VLLM_SM75_SPEC_SYNC_MODE nosync
      set_mode_default VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH 1
      ;;
    safe)
      set_mode_default ENFORCE_EAGER 1
      set_mode_default DISABLE_LOG_STATS 0
      set_mode_default VLLM_SM75_SPEC_SYNC_MODE safe
      set_mode_default VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH 0
      ;;
    *)
      die "MODE must be safe, normal, fast, or aggressive."
      ;;
  esac
}

validate_mode_kv_policy() {
  local kv=${KV_CACHE_DTYPE:-}
  local mtp=${MTP_K:-0}
  local compatible_modes=${COMPATIBLE_MODES:-safe,normal,fast}
  local mode_ok=0
  local candidate
  normalize_mode
  local normalized_modes=${compatible_modes//,/ }
  for candidate in $normalized_modes; do
    case "$candidate" in
      stable)
        candidate=safe
        ;;
      speed)
        candidate=normal
        ;;
    esac
    if [[ "$MODE" == "aggressive" ]]; then
      [[ "$candidate" == "fast" || "$candidate" == "aggressive" ]] && mode_ok=1 && break
      continue
    fi
    if [[ "$candidate" == "$MODE" ]]; then
      mode_ok=1
      break
    fi
  done
  if (( mode_ok == 0 )); then
    echo "ERROR: MODE=$MODE is not compatible with this profile." >&2
    echo "       Compatible modes: $compatible_modes" >&2
    return 1
  fi
  case "$MODE" in
    safe|normal|fast|aggressive)
      ;;
    *)
      echo "ERROR: MODE must be safe, normal, fast, or aggressive." >&2
      return 1
      ;;
  esac

  # Fast profiles are the measured serving route. Do not silently turn a
  # performance run into an eager or SDPA fallback diagnostic.
  if [[ "$MODE" == "fast" || "$MODE" == "aggressive" ]]; then
    if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
      echo "ERROR: MODE=$MODE requires CUDA Graph; ENFORCE_EAGER=1 is not allowed." >&2
      return 1
    fi
    if [[ "$kv" == turboquant_* ]]; then
      if [[ "${VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL:-1}" != "1" ]]; then
        echo "ERROR: MODE=$MODE TurboQuant requires FlashInfer prefill; set VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL=1." >&2
        return 1
      fi
      if [[ "${VLLM_TURBOQUANT_FORCE_DECODE_SDPA:-0}" == "1" || "${VLLM_TURBOQUANT_FORCE_CONTINUATION_SDPA:-0}" == "1" ]]; then
        echo "ERROR: MODE=$MODE TurboQuant does not permit a forced SDPA fallback." >&2
        return 1
      fi
    fi
  fi
}

set_sm75_runtime_env() {
  local flashqla_candidate runtime_parent cuda_candidate gcc_candidate
  export STABLE_ROOT="$RUNTIME_ROOT"
  export HOME=${RUN_HOME:-"$HOME"}
  if [[ -z "${CUDA_HOME:-}" ]]; then
    for cuda_candidate in "/usr/local/cuda-${PRIMARY_CUDA_VERSION:-12.8}" \
      /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-13 \
      /usr/local/cuda-12 /usr/local/cuda; do
      if [[ -x "$cuda_candidate/bin/nvcc" ]]; then
        CUDA_HOME="$cuda_candidate"
        break
      fi
    done
  fi
  export CUDA_HOME
  [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] ||
    die "CUDA toolkit not found; set CUDA_HOME explicitly."
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  if [[ -z "${CC:-}" ]]; then
    for gcc_candidate in "/usr/bin/gcc-${PRIMARY_GCC_MAJOR:-12}" \
      /usr/bin/gcc-15 /usr/bin/gcc-14 /usr/bin/gcc-13 /usr/bin/gcc-12; do
      if [[ -x "$gcc_candidate" ]]; then
        CC="$gcc_candidate"
        break
      fi
    done
  fi
  export CC
  if [[ -z "${CXX:-}" && -n "${CC:-}" ]]; then
    CXX=${CC/gcc/g++}
  fi
  export CXX
  if [[ -z "${CXX:-}" && -x /usr/bin/g++-12 ]]; then
    export CXX=/usr/bin/g++-12
  fi
  if [[ -z "${CUDAHOSTCXX:-}" && -n "${CXX:-}" ]]; then
    export CUDAHOSTCXX="$CXX"
  fi
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
  local selected_cuda_devices
  selected_cuda_devices=$(
    normalize_gpu_devices_to_uuids \
      "${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-$(detect_default_gpu_devices)}}"
  )
  export CUDA_VISIBLE_DEVICES="$selected_cuda_devices"
  export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
  runtime_parent=$(cd -- "$RUNTIME_ROOT/.." && pwd)
  if [[ -z "${FLASHQLA_ROOT:-}" ]]; then
    for flashqla_candidate in \
      "$RUNTIME_ROOT/.deps/FlashQLA-SM70-SM75" \
      "$MANAGER_ROOT/.deps/FlashQLA-SM70-SM75" \
      "$RUNTIME_ROOT/FlashQLA-SM70-SM75" \
      "$MANAGER_ROOT/FlashQLA-SM70-SM75" \
      "$runtime_parent/FlashQLA-SM70-SM75" \
      /opt/FlashQLA-SM70-SM75; do
      if [[ -d "$flashqla_candidate/flash_qla" ]]; then
        FLASHQLA_ROOT="$flashqla_candidate"
        break
      fi
    done
  fi
  export FLASHQLA_ROOT
  export PYTHONPATH="$RUNTIME_ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
  export PATH="$RUNTIME_ROOT/.venv/bin:${CUDA_HOME}/bin:$PATH"
  if [[ -n "${FLASHQLA_ROOT:-}" ]]; then
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
  fi
  export FLASHINFER_ENABLE_AOT=${FLASHINFER_ENABLE_AOT:-1}
  if [[ "${KV_CACHE_DTYPE:-}" == "int8_per_token_head" ]]; then
    export VLLM_INT8KV_FA_PREFILL=${VLLM_INT8KV_FA_PREFILL:-1}
    if [[ "$MODE" == "safe" ]]; then
      export VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT=${VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT:-1}
    else
      export VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT=${VLLM_INT8KV_FA_FIRST_CHUNK_DEQUANT:-0}
    fi
    export VLLM_INT8KV_FA_CONTINUATION_DEQUANT=${VLLM_INT8KV_FA_CONTINUATION_DEQUANT:-1}
    export VLLM_INT8KV_FA_CASCADE_DEQUANT=${VLLM_INT8KV_FA_CASCADE_DEQUANT:-1}
    export VLLM_INT8KV_FA_CASCADE_TILE_TOKENS=${VLLM_INT8KV_FA_CASCADE_TILE_TOKENS:-65536}
  fi
  if [[ "${KV_CACHE_DTYPE:-}" == turboquant_* ]]; then
    local tq_continuation_reserve_default=65536
    if [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
      local tq_max_model_len=${MAX_MODEL_LEN:-0}
      local tq_max_num_seqs=${MAX_NUM_SEQS:-1}
      if [[ "$tq_max_model_len" =~ ^[0-9]+$ && "$tq_max_num_seqs" =~ ^[0-9]+$ ]] \
        && (( tq_max_model_len >= 240000 )) \
        && (( tq_max_num_seqs <= 1 )); then
        # Long single-seq TQ continuation/prefix-cache lanes need the larger
        # workspace reserved up front; otherwise the first large continuation
        # grows the workspace at runtime and can trip OOM or bad split paths.
        tq_continuation_reserve_default=262144
      fi
    fi
    export VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL=${VLLM_TURBOQUANT_USE_FLASHINFER_PREFILL:-1}
    if [[ "$MODE" == "fast" || "$MODE" == "aggressive" ]]; then
      # The validated route must fail closed if FlashInfer cannot be planned;
      # otherwise a missing extension can quietly fall back to SDPA.
      export VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL=1
    else
      export VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL=${VLLM_TURBOQUANT_REQUIRE_FLASHINFER_PREFILL:-0}
    fi
    export VLLM_TURBOQUANT_FLASHINFER_BACKEND=${VLLM_TURBOQUANT_FLASHINFER_BACKEND:-fa2}
    export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE:-auto}
    export VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS=${VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE_MIN_TOKENS:-20480}
    export VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS:-$tq_continuation_reserve_default}
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=${VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE:-1}
    # SM75 MTP graphs use one fixed B=1 prefix decode per candidate. This
    # avoids the unstable B=4 materialized page-table path while remaining
    # entirely inside the captured CUDA Graph.
    export VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS=${VLLM_TURBOQUANT_CUDAGRAPH_SPEC_PREFIX_ROWS:-1}
    export VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK=${VLLM_TURBOQUANT_CONTINUATION_SDPA_Q_CHUNK:-512}
    export VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS=${VLLM_TURBOQUANT_CONTINUATION_SDPA_MAX_QK_CELLS:-16777216}
    export VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH=${VLLM_TURBOQUANT_SPEC_CONTINUATION_DECODE_FASTPATH:-1}
    if [[ -n "${VLLM_TURBOQUANT_MAX_KV_SPLITS:-}" ]]; then
      export VLLM_TURBOQUANT_MAX_KV_SPLITS
    fi
    export VLLM_TURBOQUANT_DECODE_BLOCK_KV=${VLLM_TURBOQUANT_DECODE_BLOCK_KV:-2}
  fi
  # Keep generated kernels inside this runtime tree by default. Reusing cache
  # dirs from experiment worktrees can leave absolute paths to deleted
  # environments, but explicit overrides are useful when the runtime tree is on
  # a constrained filesystem.
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$MANAGER_ROOT/torchinductor-cache"}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$MANAGER_ROOT/triton-cache"}
  export PYTHONUNBUFFERED=1
  if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]] && ! config_key_has_explicit_value VLLM_ENFORCE_STRICT_TOOL_CALLING; then
    export VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}
  fi
  if [[ -n "${REASONING_BUDGET:-}" ]]; then
    export VLLM_DEFAULT_THINKING_TOKEN_BUDGET="$REASONING_BUDGET"
  else
    unset VLLM_DEFAULT_THINKING_TOKEN_BUDGET
  fi
}

build_args() {
  local host_arg=$1

  enforce_parallelism_constraints

  VLLM_ARGS=(
    --host "$host_arg"
    --port "$PORT"
    --model "$MODEL_DIR"
    --served-model-name "$SERVED_NAME"
    --dtype half
    --tensor-parallel-size "${TP_SIZE:-2}"
    --pipeline-parallel-size "${PP_SIZE:-1}"
    --generation-config vllm
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  )

  if [[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ]]; then
    VLLM_ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
  fi
  if [[ "${PREFILL_BATCH_BARRIER:-0}" == "1" ]]; then
    VLLM_ARGS+=(--prefill-batch-barrier)
  fi

  [[ -n "${QUANTIZATION:-}" ]] && VLLM_ARGS+=(--quantization "$QUANTIZATION")
  [[ -n "${KV_CACHE_DTYPE:-}" ]] && VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    VLLM_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  else
    VLLM_ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
  fi
  [[ -n "${MAMBA_CACHE_MODE:-}" ]] && VLLM_ARGS+=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
  [[ "${ENFORCE_EAGER:-0}" == "1" ]] && VLLM_ARGS+=(--enforce-eager)
  [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]] && VLLM_ARGS+=(--no-async-scheduling)
  [[ "${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" == "1" ]] && VLLM_ARGS+=(--disable-hybrid-kv-cache-manager)
  if [[ "${DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    VLLM_ARGS+=(--no-enable-prefix-caching)
  elif [[ "${ENABLE_PREFIX_CACHING:-1}" == "1" ]]; then
    VLLM_ARGS+=(--enable-prefix-caching)
  fi
  [[ "${ENABLE_PROMPT_TOKENS_DETAILS:-1}" == "1" ]] && VLLM_ARGS+=(--enable-prompt-tokens-details)
  [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]] && VLLM_ARGS+=(--language-model-only)
  [[ "${SKIP_MM_PROFILING:-0}" == "1" ]] && VLLM_ARGS+=(--skip-mm-profiling)
  [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && VLLM_ARGS+=(--disable-custom-all-reduce)
  [[ "${DISABLE_LOG_STATS:-0}" == "1" ]] && VLLM_ARGS+=(--disable-log-stats)
  [[ -n "${ATTENTION_BACKEND:-}" ]] && VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
  if [[ -n "${REASONING_PARSER:-}" ]] && ! reasoning_parser_is_disabled; then
    VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
  fi
  [[ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && VLLM_ARGS+=(--default-chat-template-kwargs "$DEFAULT_CHAT_TEMPLATE_KWARGS")
  [[ -n "${TOOL_PARSER_PLUGIN:-}" ]] && VLLM_ARGS+=(--tool-parser-plugin "$TOOL_PARSER_PLUGIN")
  [[ -n "${TOOL_CALL_PARSER:-}" ]] && VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
  [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]] && VLLM_ARGS+=(--enable-auto-tool-choice)
  [[ -n "${ADDITIONAL_CONFIG_JSON:-}" ]] && VLLM_ARGS+=(--additional-config "$ADDITIONAL_CONFIG_JSON")
  [[ -n "${HF_OVERRIDES_JSON:-}" ]] && VLLM_ARGS+=(--hf-overrides "$HF_OVERRIDES_JSON")

  if [[ -n "${MM_LIMIT_JSON:-}" ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt "$MM_LIMIT_JSON")
  elif [[ "$MODEL_FAMILY" == qwen* && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  elif [[ "$MODEL_FAMILY" == gemma* ]]; then
    VLLM_ARGS+=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  fi

  if [[ "$MODEL_FAMILY" == qwen* && -n "${MM_LIMIT_JSON:-}" && -z "${ADDITIONAL_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}')
  fi

  if [[ -n "${CHAT_TEMPLATE_PRESET:-}" ]]; then
    local resolved_template
    if resolved_template=$(resolve_template_file "$CHAT_TEMPLATE_PRESET"); then
      CHAT_TEMPLATE_FILE="$resolved_template"
    fi
  fi
  [[ -n "${CHAT_TEMPLATE_FILE:-}" ]] && VLLM_ARGS+=(--chat-template "$CHAT_TEMPLATE_FILE")

  local decode_query_len=$((MTP_K + 1))
  local decode_max_tokens=$((MAX_NUM_SEQS * decode_query_len))
  if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
    VLLM_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
  elif (( MTP_K > 0 )); then
    VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
  fi

  local cudagraph_mode
  case "$MODE" in
    safe)
      cudagraph_mode=PIECEWISE
      ;;
    normal)
      if (( MTP_K > 0 )) || [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
        cudagraph_mode=PIECEWISE
      else
        cudagraph_mode=FULL_AND_PIECEWISE
      fi
      ;;
    fast|aggressive)
      # The validated SM75 fast route keeps the MTP decode graph full.  The
      # Mamba/GDN safety downgrade is controlled explicitly by
      # VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH in the runtime resolver.
      cudagraph_mode=FULL_AND_PIECEWISE
      ;;
    *)
      cudagraph_mode=PIECEWISE
      ;;
  esac

  if [[ -n "${COMPILATION_CONFIG_JSON:-}" ]]; then
    VLLM_ARGS+=(--compilation-config "$COMPILATION_CONFIG_JSON")
  elif [[ -n "${SPECULATIVE_CONFIG:-}" || "$MTP_K" -gt 0 ]]; then
    # Prefill stays on the piecewise compiled path.  Capturing the full
    # max-num-batched-tokens budget as a CUDA graph adds a large graph and
    # leaves long-prefill kernels to JIT during the first request.
    local capture_sizes="${decode_query_len}"
    local capture_size=$((decode_query_len * 2))
    while (( capture_size <= decode_max_tokens )); do
      capture_sizes+=",${capture_size}"
      capture_size=$((capture_size * 2))
    done
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${cudagraph_mode}\",\"cudagraph_capture_sizes\":[${capture_sizes}],\"max_cudagraph_capture_size\":${decode_max_tokens}}")
  else
    # Keep a small decode graph ladder; chunked prefills remain non-eager via
    # the piecewise compiled range and are not CUDA-graph captures.
    local capture_sizes="1"
    local capture_size=2
    local capture_max=$((MAX_NUM_SEQS * 2))
    while (( capture_size <= capture_max )); do
      capture_sizes+=",${capture_size}"
      capture_size=$((capture_size * 2))
    done
    VLLM_ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${cudagraph_mode}\",\"cudagraph_capture_sizes\":[${capture_sizes}],\"max_cudagraph_capture_size\":${capture_max}}")
  fi
}

startup_status_line() {
  local log_file=$1
  [[ -s "$log_file" ]] || {
    printf 'waiting for first log line'
    return 0
  }
  tail -n 30 "$log_file" 2>/dev/null |
    awk '
      NF {
        line=$0
      }
      END {
        if (line == "") {
          exit
        }
        worker = ""
        if (match(line, /(worker_tp[0-9]+|Worker[^ :]*|TP[0-9]+)/)) {
          worker = substr(line, RSTART, RLENGTH)
        }
        sub(/^.*(INFO|WARNING|ERROR)[^:]*:[[:space:]]*/, "", line)
        if (worker != "") {
          print "(" worker " " line ")"
        } else {
          print line
        }
      }
    ' |
    tr '\t\r\n' '   ' |
    sed -E 's/[[:space:]]+/ /g; s/^[[:space:]]+|[[:space:]]+$//g' |
    cut -c1-100
}

startup_stage() {
  local log_file=$1

  if grep -qE 'Uvicorn running|Application startup complete|Started server process' "$log_file" 2>/dev/null; then
    printf 'Starting API server'
  elif grep -qE 'Capturing CUDA graphs|CUDA graph' "$log_file" 2>/dev/null; then
    printf 'Capturing CUDA graphs'
  elif grep -qE 'Loading model|Loading weights|model weights|safetensors|shard' "$log_file" 2>/dev/null; then
    printf 'Loading model weights'
  elif grep -qE 'Initializing|init|Engine' "$log_file" 2>/dev/null; then
    printf 'Initializing engine'
  else
    printf 'Starting process'
  fi
}

startup_progress_percent() {
  local elapsed=$1
  local timeout=$2
  local log_file=$3
  local percent

  if grep -qE 'Uvicorn running|Application startup complete' "$log_file" 2>/dev/null; then
    echo 95
    return 0
  fi
  if grep -qE 'Capturing CUDA graphs|CUDA graph' "$log_file" 2>/dev/null; then
    echo 75
    return 0
  fi
  if grep -qE 'Loading model|Loading weights|model weights|safetensors|shard' "$log_file" 2>/dev/null; then
    echo 45
    return 0
  fi
  percent=$((elapsed * 90 / timeout))
  (( percent < 5 )) && percent=5
  (( percent > 90 )) && percent=90
  echo "$percent"
}

progress_bar() {
  local percent=$1
  local width=${2:-32}
  local filled empty
  filled=$((percent * width / 100))
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3s%%' "$percent"
}

render_startup_progress() {
  local log_file=$1
  local elapsed=$2
  local timeout=$3
  local frame=$4
  local percent stage hint

  percent=$(startup_progress_percent "$elapsed" "$timeout" "$log_file")
  stage=$(startup_stage "$log_file")
  hint=$(startup_status_line "$log_file")

  {
    clear
    banner
    echo "Starting service"
    echo
    printf '  Progress: %s %s\n' "$(progress_bar "$percent")" "$frame"
    printf '  Time:     %ss / %ss\n' "$elapsed" "$timeout"
    printf '  Stage:    %s\n' "$stage"
    printf '  Status:   %s\n' "$hint"
    echo
    echo "Full startup log is still written to:"
    echo "  $log_file"
  } >/dev/tty
}

wait_for_ready() {
  local log_file=$1
  local url_host=$2
  local deadline=$((SECONDS + START_TIMEOUT))
  local fatal_regex='RuntimeError|ValueError|NotImplementedError|CUDA error|OutOfMemoryError|No supported config format|EngineCore encountered a fatal error|EngineDeadError'
  local pid=""
  local start_seconds=$SECONDS
  local frames=('|' '/' '-' '\\')
  local frame=0 hint elapsed last_notice=0
  if [[ -n "${CURRENT_SERVER_PID:-}" ]]; then
    pid="$CURRENT_SERVER_PID"
  fi

  while (( SECONDS < deadline )); do
    if curl -fsS "http://${url_host}:${PORT}/health" >/dev/null 2>&1; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "OK"
        printf 'Startup complete after %ss.\n' "$((SECONDS - start_seconds))" >/dev/tty
      fi
      return 0
    fi
    if grep -E "$fatal_regex" "$log_file" >/dev/null 2>&1; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "!"
        printf 'Startup failed.\n' >/dev/tty
      fi
      return 1
    fi
    if [[ -n "$pid" && ! -d "/proc/$pid" ]]; then
      if is_tty; then
        render_startup_progress "$log_file" "$((SECONDS - start_seconds))" "$START_TIMEOUT" "!"
        printf 'Startup failed: process exited.\n' >/dev/tty
      fi
      return 1
    fi
    elapsed=$((SECONDS - start_seconds))
    hint=$(startup_status_line "$log_file")
    if is_tty; then
      render_startup_progress "$log_file" "$elapsed" "$START_TIMEOUT" "${frames[$frame]}"
      frame=$(((frame + 1) % ${#frames[@]}))
    elif (( elapsed - last_notice >= 30 )); then
      echo "Starting server... elapsed=${elapsed}s | status=$hint"
      last_notice=$elapsed
    fi
    sleep 2
  done
  if is_tty; then
    render_startup_progress "$log_file" "$START_TIMEOUT" "$START_TIMEOUT" "!"
    printf 'Startup timed out after %ss.\n' "$START_TIMEOUT" >/dev/tty
  fi
  return 2
}

cold_compile_admission_failure() {
  local log_file=$1

  [[ "${VLLM_COMPILE_PREWARM_RETRY:-0}" != "1" ]] || return 1
  [[ "${VLLM_COMPILE_PREWARM:-1}" != "0" ]] || return 1
  [[ -s "$log_file" ]] || return 1

  grep -qE 'To serve at least one request.*max seq len|estimated maximum model length is [0-9]+' "$log_file" 2>/dev/null || return 1
  grep -qE 'Compiling a graph|Cache the graph of compile range|saved AOT compiled function|Dynamo bytecode transform time' "$log_file" 2>/dev/null || return 1
  return 0
}

cold_compile_prewarm_candidates() {
  # Prewarm only needs to populate the compile cache.  Keeping the temporary
  # instance at 8K avoids allocating the full-context KV cache, especially for
  # MTP routes that also load a draft model.
  [[ "${MAX_MODEL_LEN:-0}" =~ ^[0-9]+$ && "${MAX_MODEL_LEN:-0}" -gt 8192 ]] || return 1
  printf '8192\n'
}

run_compile_prewarm() {
  local host_arg=$1
  local url_host=$2
  local prewarm_len=$3
  local original_len=$MAX_MODEL_LEN
  local original_name=$SERVED_NAME
  local original_pid=${CURRENT_SERVER_PID:-}
  local prewarm_name prewarm_safe prewarm_log prewarm_pid_file args_text ready_rc=0

  prewarm_name="${SERVED_NAME}-compile-prewarm-${prewarm_len}"
  prewarm_safe=$(printf '%s' "$prewarm_name" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
  [[ -n "$prewarm_safe" ]] || prewarm_safe="vllm-compile-prewarm"
  prewarm_log="$LOG_DIR/vllm-${prewarm_safe}-${STAMP}.log"
  prewarm_pid_file="$LOG_DIR/vllm-${prewarm_safe}.pid"

  MAX_MODEL_LEN=$prewarm_len
  SERVED_NAME=$prewarm_name
  build_args "$host_arg"
  printf -v args_text '%q ' "${VLLM_ARGS[@]}"

  {
    echo "============================================================"
    echo "$PROJECT_NAME v$VERSION compile prewarm"
    echo "Launch time: $(date '+%F %T %Z')"
    echo "Original served name: $original_name"
    echo "Original max model len: $original_len"
    echo "Prewarm max model len: $prewarm_len"
    echo "Model: $MODEL_DIR"
    echo "Mode: $MODE"
    echo "GPU devices: ${GPU_DEVICES:-}"
    echo "TP size: ${TP_SIZE:-}"
    echo "PP size: ${PP_SIZE:-1}"
    if [[ "${PP_MTP_ASYNC_AUTO_DISABLED:-0}" == "1" ]]; then
      echo "Async scheduling: disabled automatically for PP + MTP compatibility"
    fi
    if [[ "${PP_PLE_ASYNC_AUTO_DISABLED:-0}" == "1" ]]; then
      echo "Async scheduling: disabled automatically for PP + PLE compatibility"
    fi
    echo "Command: $RUNTIME_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
    echo "============================================================"
  } > "$prewarm_log"

  echo
  echo "Cold compile admission failure detected."
  echo "Running compile prewarm at max_model_len=$prewarm_len, then retrying the original $original_len context."
  echo "  Prewarm log: $prewarm_log"

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >>"$prewarm_log" 2>&1 &
  else
    nohup "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >>"$prewarm_log" 2>&1 &
  fi
  CURRENT_SERVER_PID=$!
  echo "$CURRENT_SERVER_PID" > "$prewarm_pid_file"

  wait_for_ready "$prewarm_log" "$url_host" || ready_rc=$?
  cleanup_failed_launch "$prewarm_pid_file" || true

  MAX_MODEL_LEN=$original_len
  SERVED_NAME=$original_name
  CURRENT_SERVER_PID=$original_pid
  build_args "$host_arg"

  if [[ "$ready_rc" == "0" ]]; then
    echo "Compile prewarm: OK"
    return 0
  fi

  echo "Compile prewarm failed. See: $prewarm_log" >&2
  return 1
}

smoke_test() {
  local url_host=$1
  local model_id model_output
  model_output=$("$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=30) as resp:
    data = json.load(resp)
items = data.get("data") or []
if not items:
    raise SystemExit("no model returned")
print(items[0]["id"])
PY
)
  # Runtime sitecustomize hooks may print diagnostic lines on Python startup.
  model_id=$(printf '%s\n' "$model_output" | tail -n 1)

  "$RUNTIME_ROOT/.venv/bin/python" - "$url_host" "$PORT" "$model_id" <<'PY'
import json
import sys
import urllib.request

host, port, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.load(resp)
msg = data["choices"][0]["message"]
text = (
    msg.get("content")
    or msg.get("reasoning_content")
    or msg.get("reasoning")
    or ""
)
text = str(text).strip()
if not text:
    raise SystemExit("empty smoke response")
print(text.replace("\n", " ")[:120])
PY
}

launch_server() {
  mkdir -p "$LOG_DIR"
  local safe_name log_file pid_file host_arg url_host args_text
  if [[ -z "${SERVED_NAME:-}" ]]; then
    SERVED_NAME=$(basename "$MODEL_DIR")
  fi
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  PP_SIZE=${PP_SIZE:-1}
  if [[ -z "${TP_SIZE:-}" ]]; then
    local gpu_count
    gpu_count=$(gpu_device_count "$GPU_DEVICES")
    if [[ "$PP_SIZE" =~ ^[1-9][0-9]*$ ]] && (( gpu_count % PP_SIZE == 0 )); then
      TP_SIZE=$((gpu_count / PP_SIZE))
    else
      TP_SIZE=$gpu_count
    fi
  fi
  if [[ -z "${SERVED_NAME:-}" || "$SERVED_NAME" == "." || "$SERVED_NAME" == "/" ]]; then
    echo "ERROR: Served model name is empty. Set SERVED_NAME or choose a valid checkpoint directory." >&2
    return 1
  fi
  validate_parallelism || return 1
  safe_name=$(printf '%s' "$SERVED_NAME" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
  [[ -n "$safe_name" ]] || safe_name="vllm"
  log_file="$LOG_DIR/vllm-${safe_name}-${STAMP}.log"
  pid_file="$LOG_DIR/vllm-${safe_name}.pid"

  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    host_arg="0.0.0.0"
    url_host="127.0.0.1"
  else
    host_arg="127.0.0.1"
    url_host="127.0.0.1"
  fi

  build_args "$host_arg"
  printf -v args_text '%q ' "${VLLM_ARGS[@]}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo
    echo "DRY RUN"
    echo "Environment:"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "  VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}"
    echo "  VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
    echo "  VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
    echo "  VLLM_DEFAULT_THINKING_TOKEN_BUDGET=${VLLM_DEFAULT_THINKING_TOKEN_BUDGET:-}"
    echo "Command:"
    echo "  $RUNTIME_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
    return 0
  fi

  check_checkpoint_mmap_policy || return 1
  warn_display_gpu_occupancy || true

  {
    echo "============================================================"
    echo "$PROJECT_NAME v$VERSION"
    echo "Runtime identity: $RUNTIME_IDENTITY"
    echo "Base vLLM: $BASE_VLLM_VERSION"
    echo "Validated CUDA/Torch: CUDA $VALIDATED_CUDA_VERSION / torch $VALIDATED_TORCH_VERSION"
    echo "Reference NVIDIA driver: $VALIDATED_NVIDIA_DRIVER_VERSION"
    echo "Launch time: $(date '+%F %T %Z')"
    echo "Served name: $SERVED_NAME"
    echo "Model: $MODEL_DIR"
    echo "Profile: ${PROFILE:-manual}"
    echo "Mode: $MODE"
    echo "GPU devices: ${GPU_DEVICES:-}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
    echo "TP size: ${TP_SIZE:-}"
    echo "Port: $PORT"
    echo "Scope: $SERVICE_SCOPE"
    echo "MTP graph policy: VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
    echo "TQ diagnostics: $(current_tq_diagnostics_label)"
    echo "Strict tool calling: VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
    echo "Command: $RUNTIME_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server $args_text"
    echo "============================================================"
  } > "$log_file"

  echo
  echo "Starting server..."
  echo "  Log: $log_file"
  echo "  Mode: $MODE"
  echo "  MTP graph policy: VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}"
  if [[ "${PP_MTP_ASYNC_AUTO_DISABLED:-0}" == "1" ]]; then
    echo "  Async scheduling: disabled automatically for PP + MTP compatibility"
  fi
  if [[ "${PP_PLE_ASYNC_AUTO_DISABLED:-0}" == "1" ]]; then
    echo "  Async scheduling: disabled automatically for PP + PLE compatibility"
  fi
  echo "  TQ diagnostics: $(current_tq_diagnostics_label)"
  echo "  Strict tool calling: VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
  echo "  Served name: $SERVED_NAME"
  echo "  Model: $MODEL_DIR"
  echo "  Bind: $host_arg:$PORT"

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >>"$log_file" 2>&1 &
  else
    nohup "$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" >>"$log_file" 2>&1 &
  fi
  CURRENT_SERVER_PID=$!
  echo "$CURRENT_SERVER_PID" > "$pid_file"

  local ready_rc=0
  wait_for_ready "$log_file" "$url_host" || ready_rc=$?
  if [[ "$ready_rc" != "0" ]]; then
    local prewarm_candidates prewarm_len retry_rc prewarm_ok=0
    if cold_compile_admission_failure "$log_file"; then
      prewarm_candidates=$(cold_compile_prewarm_candidates "$log_file" || true)
      if [[ -n "$prewarm_candidates" ]]; then
        cleanup_failed_launch "$pid_file"
        while IFS= read -r prewarm_len; do
          [[ -n "$prewarm_len" ]] || continue
          echo "Trying compile prewarm at max_model_len=$prewarm_len..."
          if run_compile_prewarm "$host_arg" "$url_host" "$prewarm_len"; then
            prewarm_ok=1
            break
          fi
        done <<< "$prewarm_candidates"
        if (( prewarm_ok )); then
          echo
          echo "Retrying original launch after compile prewarm..."
          local old_stamp=$STAMP
          STAMP=$(date +%Y%m%d-%H%M%S)
          VLLM_COMPILE_PREWARM_RETRY=1 launch_server
          retry_rc=$?
          STAMP=$old_stamp
          unset VLLM_COMPILE_PREWARM_RETRY
          return "$retry_rc"
        fi
      fi
    fi
    echo
    echo "START FAILED"
    echo "Log: $log_file"
    tail -n 120 "$log_file" || true
    echo "Cleaning up failed server processes..."
    cleanup_failed_launch "$pid_file"
    restore_overcommit_memory || true
    return 1
  fi

  echo "Health check: OK"
  if [[ "${SKIP_STARTUP_SMOKE:-0}" == "1" ]]; then
    restore_overcommit_memory || true

    local api_local="http://127.0.0.1:${PORT}/v1"
    local api_lan=""
    if [[ "$SERVICE_SCOPE" == "lan" ]]; then
      api_lan="http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/v1"
    fi
    LAST_PID_FILE="$pid_file"
    LAST_LOG_FILE="$log_file"
    LAST_API_LOCAL="$api_local"
    LAST_API_LAN="$api_lan"
    LAST_SMOKE_OUTPUT="skipped"
    save_manager_state

    echo
    echo "START OK"
    echo "Smoke response: skipped"
    echo "PID file: $pid_file"
    echo "Log: $log_file"
    echo "Local API: $api_local"
    if [[ -n "$api_lan" ]]; then
      echo "LAN API:   $api_lan"
    fi

    if is_tty; then
      show_launch_status
    fi
    return 0
  fi

  local smoke_output
  echo "Running smoke test..."
  if ! smoke_output=$(smoke_test "$url_host" 2>&1); then
    echo
    echo "SMOKE FAILED"
    echo "$smoke_output"
    echo "Log: $log_file"
    echo "Cleaning up failed server processes..."
    cleanup_failed_launch "$pid_file"
    restore_overcommit_memory || true
    return 1
  fi

  restore_overcommit_memory || true

  local api_local="http://127.0.0.1:${PORT}/v1"
  local api_lan=""
  if [[ "$SERVICE_SCOPE" == "lan" ]]; then
    api_lan="http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/v1"
  fi
  LAST_PID_FILE="$pid_file"
  LAST_LOG_FILE="$log_file"
  LAST_API_LOCAL="$api_local"
  LAST_API_LAN="$api_lan"
  LAST_SMOKE_OUTPUT="$smoke_output"
  save_manager_state

  echo
  echo "START OK"
  echo "Smoke response: $smoke_output"
  echo "PID file: $pid_file"
  echo "Log: $log_file"
  echo "Local API: $api_local"
  if [[ -n "$api_lan" ]]; then
    echo "LAN API:   $api_lan"
  fi

  if is_tty; then
    show_launch_status
  fi
}

set_overcommit_memory_one() {
  if [[ -w /proc/sys/vm/overcommit_memory ]]; then
    echo 1 >/proc/sys/vm/overcommit_memory
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo sysctl -w vm.overcommit_memory=1 >/dev/null
    return 0
  fi
  return 1
}

restore_overcommit_memory() {
  local value=${OVERCOMMIT_RESTORE_VALUE:-}
  [[ "${OVERCOMMIT_CHANGED:-0}" == "1" && -n "$value" ]] || return 0

  if [[ -w /proc/sys/vm/overcommit_memory ]]; then
    echo "$value" >/proc/sys/vm/overcommit_memory
  elif command -v sudo >/dev/null 2>&1; then
    sudo sysctl -w "vm.overcommit_memory=$value" >/dev/null
  else
    echo "WARNING: could not restore vm.overcommit_memory=$value; missing sudo/root permission." >&2
    return 1
  fi

  OVERCOMMIT_CHANGED=0
  OVERCOMMIT_RESTORE_VALUE=""
  echo "INFO: Restored vm.overcommit_memory=$value."
}

trap 'restore_overcommit_memory >/dev/null 2>&1 || true' EXIT

meminfo_kib() {
  local key=$1
  awk -v key="$key:" '$1 == key { print $2; exit }' /proc/meminfo 2>/dev/null
}

commit_headroom_bytes() {
  local limit_kib committed_kib
  limit_kib=$(meminfo_kib CommitLimit)
  committed_kib=$(meminfo_kib Committed_AS)
  [[ "$limit_kib" =~ ^[0-9]+$ && "$committed_kib" =~ ^[0-9]+$ ]] || return 1
  if (( limit_kib <= committed_kib )); then
    echo 0
  else
    echo $(((limit_kib - committed_kib) * 1024))
  fi
}

bytes_to_gib() {
  local bytes=${1:-0}
  awk -v bytes="$bytes" 'BEGIN { printf "%.2f GiB", bytes / 1024 / 1024 / 1024 }'
}

largest_safetensors_file() {
  find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -printf '%s\t%p\n' 2>/dev/null |
    sort -nr |
    head -n 1
}

check_checkpoint_mmap_policy() {
  local overcommit largest_line largest_bytes largest_path headroom_bytes answer

  [[ -r /proc/sys/vm/overcommit_memory ]] || return 0
  overcommit=$(cat /proc/sys/vm/overcommit_memory 2>/dev/null || true)
  [[ "$overcommit" == "0" ]] || return 0

  largest_line=$(largest_safetensors_file)
  [[ -n "$largest_line" ]] || return 0
  largest_bytes=${largest_line%%$'\t'*}
  largest_path=${largest_line#*$'\t'}
  [[ "$largest_bytes" =~ ^[0-9]+$ ]] || return 0
  headroom_bytes=$(commit_headroom_bytes 2>/dev/null || true)
  [[ "$headroom_bytes" =~ ^[0-9]+$ ]] || return 0
  (( largest_bytes > headroom_bytes )) || return 0

  echo
  echo "Checkpoint mmap preflight"
  echo "  Largest safetensors: $(bytes_to_gib "$largest_bytes")"
  echo "  Commit headroom:     $(bytes_to_gib "$headroom_bytes")"
  echo "  File:                $largest_path"
  echo
  echo "This host is using vm.overcommit_memory=0, and the largest checkpoint"
  echo "file is bigger than the current commit headroom. vLLM may fail with:"
  echo "  unable to mmap ... Cannot allocate memory (12)"
  echo

  if is_tty; then
    answer=$(read_line_with_esc "Enable vm.overcommit_memory=1 with sudo now? [y/N]: ") || return 1
    case "$answer" in
      y|Y)
        ;;
      *)
        echo "Start cancelled. vm.overcommit_memory was not changed." >&2
        return 1
        ;;
    esac
  elif ! [[ -w /proc/sys/vm/overcommit_memory ]]; then
    echo "ERROR: checkpoint mmap needs vm.overcommit_memory=1, but non-interactive launcher cannot prompt for sudo." >&2
    return 1
  fi

  if set_overcommit_memory_one; then
    OVERCOMMIT_RESTORE_VALUE="$overcommit"
    OVERCOMMIT_CHANGED=1
    echo "INFO: Enabled vm.overcommit_memory=1 temporarily for this launch."
    return 0
  fi

  echo "ERROR: could not enable vm.overcommit_memory=1. The current user needs sudo/root permission." >&2
  return 1
}

prepare_runtime_defaults() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "ERROR: MODEL_DIR is required. Choose item 1 first." >&2
    return 1
  fi
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory does not exist: $MODEL_DIR" >&2
    return 1
  fi
  MODEL_FAMILY=${MODEL_FAMILY:-$(guess_model_family "$MODEL_DIR")}
  SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_DIR")}
  TEMPLATE_DIR=${TEMPLATE_DIR:-"$PROFILE_DIR/templates"}
  GPU_DEVICES=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  PP_SIZE=${PP_SIZE:-1}
  if [[ -z "${TP_SIZE:-}" ]]; then
    local gpu_count
    gpu_count=$(gpu_device_count "$GPU_DEVICES")
    if [[ "$PP_SIZE" =~ ^[1-9][0-9]*$ ]] && (( gpu_count % PP_SIZE == 0 )); then
      TP_SIZE=$((gpu_count / PP_SIZE))
    else
      TP_SIZE=$gpu_count
    fi
  fi
  QUANTIZATION=${QUANTIZATION:-$(guess_quantization "$MODEL_DIR")}
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(default_context_tokens)}
  GPU_UTIL=${GPU_UTIL:-$(default_gpu_util)}
  MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  PREFILL_BATCH_BARRIER=${PREFILL_BATCH_BARRIER:-0}
  MTP_K=${MTP_K:-0}
  PORT=${PORT:-8000}
  MODE=${MODE:-normal}
  normalize_mode
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}
  normalize_message_type_defaults
  apply_prefix_cache_defaults
  ENABLE_AUTO_TOOL_CHOICE=$(normalize_bool "${ENABLE_AUTO_TOOL_CHOICE:-0}")
  apply_family_reasoning_defaults
  if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    if ! config_key_has_explicit_value TOOL_CALL_PARSER; then
      TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}
    fi
    if ! config_key_has_explicit_value VLLM_ENFORCE_STRICT_TOOL_CALLING; then
      VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}
    fi
  fi
  validate_mode_kv_policy
  validate_parallelism || return 1
  enforce_parallelism_constraints
}

collect_config_env() {
  local profile_file
  if profile_file=$(resolve_profile_file); then
    apply_profile_overrides "$profile_file"
  fi
  prepare_runtime_defaults || die "Invalid runtime configuration."
}

print_review() {
  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  local message_type=text-only
  [[ -n "${MM_LIMIT_JSON:-}" ]] && message_type=text+image
  local kv_cache_memory_label=${KV_CACHE_MEMORY_BYTES:-auto}
  local gpu_util_label=$GPU_UTIL
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    gpu_util_label="$GPU_UTIL (fallback, ignored)"
  fi

  cat <<EOF
Launch summary:
  Fork release:         v$VERSION
  Runtime identity:     $RUNTIME_IDENTITY
  Model directory:      $MODEL_DIR
  Served name:          $SERVED_NAME
  Model family:         $MODEL_FAMILY
  vLLM --quantization:  ${QUANTIZATION:-auto}
  W/A type:             $(guess_precision_scheme "$MODEL_DIR" "${QUANTIZATION:-}")
  GPU devices:          ${GPU_DEVICES:-$(detect_default_gpu_devices)}
  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}
  TP / PP:              ${TP_SIZE:-} / ${PP_SIZE:-1}
  KV precision:         ${KV_CACHE_DTYPE:-fp16}
  TQ diagnostics:       $(current_tq_diagnostics_label)
  Prefix cache:         $(current_prefix_cache_label)
  Mamba cache mode:     ${MAMBA_CACHE_MODE:-auto}
  Context tokens:       $MAX_MODEL_LEN
  KV cache memory:      $kv_cache_memory_label
  GPU util:             $gpu_util_label
  Max batched tokens:   $MAX_BATCHED_TOKENS
  Max sequences:        $MAX_NUM_SEQS
  Prefill threshold:    ${LONG_PREFILL_TOKEN_THRESHOLD:-auto}
  Prefill batch barrier: ${PREFILL_BATCH_BARRIER:-0}
  MTP tokens:           $MTP_K
  Message type:         $message_type
  Chat template:        $(current_template_label)
  Reasoning default:    $(current_reasoning_label)
  Tool calling:         $(current_tool_calling_label)
  Prompt details:       ${ENABLE_PROMPT_TOKENS_DETAILS:-1}
  Mode:                 $MODE
  MTP graph policy:     VLLM_SM75_SPEC_SYNC_MODE=${VLLM_SM75_SPEC_SYNC_MODE:-auto}, VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=${VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH:-0}
  Async scheduling:     $(if [[ "${NO_ASYNC_SCHEDULING:-0}" == "1" ]]; then printf 'disabled'; else printf 'enabled'; fi)$(if [[ "${PP_MTP_ASYNC_AUTO_DISABLED:-0}" == "1" ]]; then printf ' (automatic PP + MTP safeguard)'; fi)
  Strict tool calling:  VLLM_ENFORCE_STRICT_TOOL_CALLING=${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}
  Port:                 $PORT
  Scope:                $SERVICE_SCOPE
EOF
  echo
}

start_configured_service() {
  if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "Set item 1: weight/checkpoint directory first."
    pause_enter
    return 0
  fi

  if ! prepare_runtime_defaults; then
    pause_enter
    return 0
  fi

  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  if current_service_info >/dev/null; then
    confirm_restart_existing || {
      pause_enter
      return 0
    }
  else
    confirm_start || return 0
  fi

  if launch_server; then
    echo
    echo "Press Enter to return to the main menu. The service will keep running."
    pause_enter
  else
    echo
    echo "Returned to main menu."
    pause_enter
  fi
}

menu_value() {
  local value=${1:-}
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '<unset>'
  fi
}

render_main_menu_item() {
  local idx=$1
  local current=$2
  local text=$3

  if (( idx == current )); then
    printf ' > %s\n' "$text"
  else
    printf '   %s\n' "$text"
  fi
}

render_main_menu() {
  local current=${1:-1}
  local gpu_devices tp_size pp_size kv_cache_memory_label gpu_util_label
  gpu_devices=${GPU_DEVICES:-$(detect_default_gpu_devices)}
  tp_size=${TP_SIZE:-$(gpu_device_count "$gpu_devices")}
  pp_size=${PP_SIZE:-1}
  kv_cache_memory_label=${KV_CACHE_MEMORY_BYTES:-auto}
  gpu_util_label=${GPU_UTIL:-$(default_gpu_util)}
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    gpu_util_label="${gpu_util_label} (fallback)"
  fi

  if is_tty; then
    clear >/dev/tty 2>/dev/null || true
  fi
  banner
  echo "Main menu"
  echo
  render_service_status
  render_main_menu_item 1 "$current" "1. Weight directory: $(menu_value "${MODEL_DIR:-}")"
  render_main_menu_item 2 "$current" "2. Profile:          $(current_profile_label)"
  printf '     Model family:     %s\n' "$(menu_value "${MODEL_FAMILY:-}")"
  printf '     Served name:      %s\n' "$(menu_value "${SERVED_NAME:-}")"
  printf '     vLLM quant:       %s\n' "$(menu_value "${QUANTIZATION:-auto}")"
  printf '     W/A type:         %s\n' "$(menu_value "$(guess_precision_scheme "${MODEL_DIR:-}" "${QUANTIZATION:-}")")"
  printf '     KV precision:     %s\n' "$(menu_value "${KV_CACHE_DTYPE:-fp16}")"
  printf '     Prefix cache:     %s\n' "$(menu_value "$(current_prefix_cache_label)")"
  printf '     Context tokens:   %s\n' "$(menu_value "${MAX_MODEL_LEN:-$(default_context_tokens)}")"
  printf '     KV cache memory:  %s\n' "$(menu_value "$kv_cache_memory_label")"
  printf '     GPU util:         %s\n' "$(menu_value "$gpu_util_label")"
  printf '     Batch tokens:     %s\n' "$(menu_value "${MAX_BATCHED_TOKENS:-2048}")"
  printf '     Max sequences:    %s\n' "$(menu_value "${MAX_NUM_SEQS:-1}")"
  printf '     MTP tokens:       %s\n' "$(menu_value "${MTP_K:-0}")"
  printf '     Message type:     %s\n' "$(menu_value "${MESSAGE_TYPE:-text-only}")"
  printf '     Chat template:    %s\n' "$(menu_value "$(current_template_label)")"
  printf '     Reasoning:        %s\n' "$(menu_value "$(current_reasoning_label)")"
  printf '     Tool calling:     %s\n' "$(menu_value "$(current_tool_calling_label)")"
  render_main_menu_item 3 "$current" "3. GPU/TP/PP setting: $(menu_value "$gpu_devices") / TP $(menu_value "$tp_size") / PP $(menu_value "$pp_size")"
  render_main_menu_item 4 "$current" "4. Launch mode:      ${MODE:-normal}"
  render_main_menu_item 5 "$current" "5. Port:             ${PORT:-8000}"
  render_main_menu_item 6 "$current" "6. Service scope:    $(current_scope_label)"
  render_main_menu_item 7 "$current" "7. Help"
  render_main_menu_item 8 "$current" "8. Start service"
  render_main_menu_item 9 "$current" "9. Stop service"
  render_main_menu_item 0 "$current" "0. Exit"
  echo
  echo "Use Up/Down, Enter to select. Number keys jump directly. Esc/0 exits."
  echo "Profile presets are optional. They only fill editable runtime parameters."
  echo
}

read_main_menu_choice() {
  local current=$1
  local timeout=${2:-}
  local key seq selected_index

  if [[ -n "$timeout" ]]; then
    IFS= read -rsn1 -t "$timeout" key </dev/tty || return 124
  else
    IFS= read -rsn1 key </dev/tty || return 1
  fi
  printf '\n' >/dev/tty

  [[ "$key" == $'\x04' ]] && return 1
  if [[ -z "$key" ]]; then
    printf '%s\n' "$current"
    return 0
  fi

  if [[ "$key" == $'\x1b' ]]; then
    read -rsn2 -t 0.1 seq </dev/tty || true
    if [[ -z "$seq" ]]; then
      return 1
    fi
    case "$seq" in
      "[A")
        if (( current > 1 )); then
          current=$((current - 1))
        else
          current=0
        fi
        printf '__INDEX__:%s\n' "$current"
        return 0
        ;;
      "[B")
        if (( current == 0 )); then
          current=1
        elif (( current < 9 )); then
          current=$((current + 1))
        else
          current=0
        fi
        printf '__INDEX__:%s\n' "$current"
        return 0
        ;;
    esac
    return 0
  fi

  if [[ "$key" =~ ^[0-9]$ ]]; then
    printf '%s\n' "$key"
    return 0
  fi

  selected_index="$key"
  printf '%s\n' "$selected_index"
}

service_manager() {
  local choice menu_idx=1 refresh_timeout rc
  load_manager_state
  MODE=${MODE:-normal}
  PORT=${PORT:-8000}
  SERVICE_SCOPE=${SERVICE_SCOPE:-local}

  while true; do
    render_main_menu "$menu_idx"
    printf 'Select [0-9]: ' >/dev/tty
    refresh_timeout=""
    if service_has_live_kv_cache_usage; then
      refresh_timeout=${STATUS_REFRESH_SECONDS:-30}
    fi
    set +e
    choice=$(read_main_menu_choice "$menu_idx" "$refresh_timeout")
    rc=$?
    set -e
    if (( rc == 124 )); then
      continue
    fi
    if (( rc != 0 )); then
      exit 0
    fi
    case "$choice" in
      __INDEX__:*)
        menu_idx=${choice#__INDEX__:}
        continue
        ;;
    esac
    case "$choice" in
      1)
        select_weight_dir
        ;;
      2)
        select_profile_preset
        ;;
      3)
        select_gpu_devices_menu
        ;;
      4)
        select_mode_menu
        ;;
      5)
        input_port_menu
        ;;
      6)
        select_scope_menu
        ;;
      7)
        show_help
        ;;
      8)
        start_configured_service
        ;;
      9)
        stop_service
        ;;
      0|q|Q|quit|exit)
        exit 0
        ;;
      "")
        ;;
      *)
        echo "Unknown choice: $choice"
        pause_enter
        ;;
    esac
  done
}

has_arg() {
  local want=$1
  shift || true
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$want" ]] && return 0
  done
  return 1
}

run_start_flow() {
  collect_config_env
  apply_mode
  set_sm75_runtime_env
  START_TIMEOUT=${START_TIMEOUT:-900}
  print_review
  if [[ "${PRINT_CONFIG:-0}" == "1" ]] || has_arg "--print-config" "$@"; then
    return 0
  fi
  launch_server
}

main() {
  cd "$MANAGER_ROOT"

  if has_arg "--help" "$@" || has_arg "-h" "$@"; then
    show_help
    exit 0
  fi

  parse_launcher_args "$@"
  register_env_config_overrides
  apply_launcher_path_defaults
  NON_INTERACTIVE=$(normalize_bool "${NON_INTERACTIVE:-0}")
  PRINT_CONFIG=$(normalize_bool "${PRINT_CONFIG:-0}")

  if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
    banner
    die ".venv is missing under RUNTIME_ROOT=$RUNTIME_ROOT. Run ./build.sh first or set RUNTIME_ROOT."
  fi

  mkdir -p "$LOG_DIR"

  if [[ "${NON_INTERACTIVE:-0}" == "1" || "${PRINT_CONFIG:-0}" == "1" || ! -t 0 ]]; then
    run_start_flow "$@"
  else
    service_manager
  fi
}

main "$@"
