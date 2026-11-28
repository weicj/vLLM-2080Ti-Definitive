#!/usr/bin/env bash
# Shared by build.sh and the host-side launcher.
VLLM_DOCKER_COMMAND=${VLLM_DOCKER_COMMAND:-docker}
if [[ "${VLLM_RUNTIME_BACKEND:-docker}" != docker ]]; then
  printf 'This branch uses Docker. Use the 0.2.x release line for native builds.\n' >&2
  exit 1
fi
VLLM_DOCKER_IMAGE=${VLLM_DOCKER_IMAGE:-vllm-2080ti:${FORK_RELEASE}-vllm${DOCKER_VLLM_VERSION}}

docker_runtime_options() {
  local hf_dir hf_hub_dir var
  hf_dir=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
  hf_hub_dir=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$hf_dir/hub}}
  mkdir -p -- "$hf_dir" "$hf_hub_dir" || return 1
  DOCKER_RUNTIME_OPTIONS=(
    --network host
    --user "$(id -u):$(id -g)"
    --workdir "$MANAGER_ROOT"
    --env "HOME=/tmp"
    --env "HF_HOME=$hf_dir"
    --env "HF_HUB_CACHE=$hf_hub_dir"
  )
  docker_runtime_mount "$hf_dir" || return 1
  if [[ "$hf_hub_dir" != "$hf_dir/"* && "$hf_hub_dir" != "$hf_dir" ]]; then
    docker_runtime_mount "$hf_hub_dir" || return 1
  fi
  for var in HF_ENDPOINT HF_TOKEN HF_HUB_OFFLINE HF_HUB_DOWNLOAD_TIMEOUT \
      HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy \
      PYTHONHASHSEED PYTHONUNBUFFERED CUDA_DEVICE_ORDER; do
    if [[ -v "$var" ]]; then
      DOCKER_RUNTIME_OPTIONS+=(--env "$var")
    fi
  done
  # Pass exported vLLM settings by name so credentials are not printed in argv.
  while IFS= read -r var; do
    case "$var" in
      VLLM_USE_V2_MODEL_RUNNER|VLLM_SM75_SPEC_SYNC_MODE|VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH|VLLM_KV_CACHE_LAYOUT|VLLM_DOCKER_*|VLLM_RUNTIME_BACKEND) ;;
      VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC|VLLM_USE_BREAKABLE_CUDAGRAPH)
        DOCKER_RUNTIME_OPTIONS+=(--env "$var=${!var}") ;;
      VLLM_*) DOCKER_RUNTIME_OPTIONS+=(--env "$var") ;;
    esac
  done < <(compgen -e)
}

docker_runtime_mount() {
  local path=$1 mode=${2:-}
  local source
  [[ "$path" == /* ]] || path="$PWD/$path"
  source=$(realpath -e -- "$path") || return 1
  # Docker parses --mount as CSV, including paths with commas or quotes.
  local option="\"type=bind\",\"source=${source//\"/\"\"}\",\"target=${path//\"/\"\"}\""
  DOCKER_RUNTIME_OPTIONS+=(--mount "${option}${mode:+,$mode}")
}

docker_server_options() {
  local draft file cache_dir gpu_uuids gpu uuid layout spec_method
  local -a indices=() uuids=()
  docker_runtime_options || return 1
  layout=${VLLM_KV_CACHE_LAYOUT:-}
  if [[ -z "$layout" ]]; then
    spec_method=$(effective_speculative_method)
    if is_docker_tp4_t10_qwen_profile &&
       [[ "${KV_CACHE_DTYPE:-}" != turboquant_* || "$spec_method" != dflash ]]; then
      # FlashInfer can split layer-contiguous hybrid KV pages.
      layout=LBNHC
    elif [[ "$spec_method" == dflash ]]; then
      # Keep the target and draft page sizes independent.
      layout=BLHNC
    fi
  fi
  if [[ -n "$layout" ]]; then
    DOCKER_RUNTIME_OPTIONS+=(--env "VLLM_KV_CACHE_LAYOUT=$layout")
  fi
  if [[ ! -v VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC &&
        "${NCCL_P2P_DISABLE:-0}" != 1 &&
        "${CUSTOM_ALL_REDUCE_MODE:-}" != off &&
        "${DISABLE_CUSTOM_ALL_REDUCE:-0}" != 1 ]] &&
     is_docker_tp4_t10_qwen_profile && ! docker_t10_220k_prefix_cache &&
     recommend_gpu_rank_order "$GPU_DEVICES" "$TP_SIZE" "${PP_SIZE:-1}" >/dev/null 2>&1; then
    DOCKER_RUNTIME_OPTIONS+=(--env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1)
  fi
  IFS=, read -r -a indices <<<"$GPU_DEVICES"
  for gpu in "${indices[@]}"; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id "$gpu") || return 1
    [[ "$uuid" == GPU-* ]] || return 1
    uuids+=("$uuid")
  done
  gpu_uuids=$(IFS=,; printf '%s' "${uuids[*]}")
  DOCKER_RUNTIME_OPTIONS+=(
    --ipc=host --init
    --gpus "\"device=$gpu_uuids\""
    --env "CUDA_VISIBLE_DEVICES=$gpu_uuids"
    --env "HF_ENDPOINT=${HF_ACTIVE_ENDPOINT:-${HF_ENDPOINT:-$HF_OFFICIAL_ENDPOINT}}"
  )
  cache_dir="$RUNTIME_ROOT/.cache/vllm-docker-cu130-$DOCKER_VLLM_VERSION"
  mkdir -p -- "$cache_dir" || return 1
  docker_runtime_mount "$cache_dir" || return 1
  DOCKER_RUNTIME_OPTIONS+=(--env "XDG_CACHE_HOME=$cache_dir"
    --env "VLLM_CACHE_ROOT=$cache_dir/vllm"
    --env "FLASHINFER_WORKSPACE_BASE=$cache_dir"
    --env "TORCHINDUCTOR_CACHE_DIR=$cache_dir/torchinductor"
    --env "TRITON_CACHE_DIR=$cache_dir/triton")
  local nccl_var
  for nccl_var in NCCL_P2P_DISABLE NCCL_P2P_LEVEL NCCL_DEBUG NCCL_DEBUG_SUBSYS PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF; do
    if [[ -v "$nccl_var" ]]; then
      DOCKER_RUNTIME_OPTIONS+=(--env "$nccl_var=${!nccl_var}")
    fi
  done

  docker_runtime_mount "$MODEL_DIR" readonly || return 1
  draft=$(effective_speculative_model)
  if [[ -n "$draft" && -d "$draft" && "$draft" != "$MODEL_DIR" ]]; then
    docker_runtime_mount "$draft" readonly || return 1
  fi
  for file in "${CHAT_TEMPLATE_FILE:-}" "${TOOL_PARSER_PLUGIN:-}" "${GENERATION_CONFIG:-}"; do
    if [[ -n "$file" && -e "$file" ]]; then
      docker_runtime_mount "$file" readonly || return 1
    fi
  done
  if [[ -n "${KV_DISK_CACHE_DIR:-}" ]]; then
    docker_runtime_mount "$KV_DISK_CACHE_DIR" || return 1
  fi
}

docker_stop_service() {
  local pid_file=$1 container
  container=$(<"${pid_file}.container")
  [[ "$container" =~ ^[0-9a-f]{64}$ ]] || return 1
  if "$VLLM_DOCKER_COMMAND" container inspect "$container" >/dev/null 2>&1; then
    "$VLLM_DOCKER_COMMAND" stop --time 10 "$container" >/dev/null 2>&1 || true
    "$VLLM_DOCKER_COMMAND" rm --force "$container" >/dev/null || return 1
  else
    "$VLLM_DOCKER_COMMAND" info >/dev/null 2>&1 || return 1
  fi
  rm -f -- "$pid_file" "${pid_file}.container"
}
