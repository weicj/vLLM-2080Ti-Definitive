#!/usr/bin/env bash
set -euo pipefail

# Reproducible long-output comparison server used by the public benchmark video.
# MODE=baseline keeps CUDA Graph but removes MTP/FlashQLA/FlashInfer sampler;
# MODE=optimized uses the SM75 MTP3 + FlashQLA + FlashInfer stack.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${MODE:-optimized}
PORT=${PORT:-18087}
MODEL_PATH=${MODEL_PATH:-/home/ubuntu/storage/llm/Qwen/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP}
SERVED_NAME=${SERVED_NAME:-qwen27b-video-${MODE}}
LOG_FILE=${LOG_FILE:-$ROOT/test-results/video-${MODE}.server.log}
GPU_IDS=${CUDA_VISIBLE_DEVICES:-6,7}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTHONPATH="$ROOT/flashinfer-068-site:$ROOT:$ROOT/.deps/FlashQLA-SM70-SM75${PYTHONPATH:+:$PYTHONPATH}"
export FLASHQLA_ROOT="$ROOT/.deps/FlashQLA-SM70-SM75"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/cache-video-${MODE}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/cache-video-${MODE}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/cache-video-${MODE}/triton}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export CC=${CC:-/usr/bin/gcc-11}
export CXX=${CXX:-/usr/bin/g++-11}
export CUDAHOSTCXX=${CUDAHOSTCXX:-/usr/bin/g++-11}

cmd=(
  "$ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1 --port "$PORT"
  --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
  --dtype half --trust-remote-code --generation-config vllm
  --quantization awq_marlin --tensor-parallel-size 2
  --gpu-memory-utilization 0.92 --max-model-len "$MAX_MODEL_LEN"
  --enable-chunked-prefill --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --mamba-cache-mode align --disable-custom-all-reduce
  --disable-log-stats --reasoning-parser qwen3
  --limit-mm-per-prompt '{"image":4,"video":0,"audio":0}'
)

if [[ "$ENFORCE_EAGER" == 1 ]]; then
  cmd+=(--enforce-eager)
fi
if [[ "$SKIP_MM_PROFILING" == 1 ]]; then
  cmd+=(--skip-mm-profiling)
fi

if [[ "$MODE" == optimized ]]; then
  export FLASH_QLA_LEGACY_PREBUILT="$ROOT/cache-flashqla068/flash_qla_legacy_gdn/flash_qla_legacy_gdn.so"
  export VLLM_SM75_SPEC_SYNC_MODE=safe
  export VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0
  export VLLM_USE_FLASHINFER_SAMPLER=1
  export FLASHINFER_ENABLE_AOT=1
  cmd+=(
    --attention-backend FLASHINFER
    --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
  )
else
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export FLASHINFER_ENABLE_AOT=0
  cmd+=(
    --attention-backend TRITON_ATTN
    --additional-config '{"gdn_prefill_backend":"triton"}'
    --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}'
  )
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$ROOT"
echo "[$(date '+%F %T')] MODE=$MODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PORT=$PORT" >> "$LOG_FILE"
echo "[$(date '+%F %T')] ${cmd[*]}" >> "$LOG_FILE"
exec taskset -c "${CPU_AFFINITY:-16-31,48-63}" "${cmd[@]}" >> "$LOG_FILE" 2>&1
