#!/bin/bash
set -euo pipefail

# Portable startup for the Qwen3.8-27B GPTQ INT4 FP16-KV MTP3 route.
# Override MODEL_DIR (checkpoint) and RUNTIME_ROOT (runtime tree) as needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT}"
MODEL_DIR="${MODEL_DIR:-}"
LOG_DIR="${LOG_DIR:-$RUNTIME_ROOT/run-logs}"

if [[ -z "$MODEL_DIR" ]]; then
  echo "MODEL_DIR must be set to the Qwen3.8-27B GPTQ INT4 checkpoint." >&2
  exit 1
fi
if [[ ! -x "$RUNTIME_ROOT/.venv/bin/python" ]]; then
  echo "Runtime venv not found at $RUNTIME_ROOT/.venv; set RUNTIME_ROOT." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

"$RUNTIME_ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$MODEL_DIR" \
  --served-model-name qwen38-gptq-fp16kv-128K-mtp3-text-only \
  --dtype half \
  --tensor-parallel-size 2 \
  --generation-config vllm \
  --max-model-len 128000 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --skip-mm-profiling \
  --disable-log-stats \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  2>&1 | tee "$LOG_DIR/vllm-qwen38-gptq-fp16kv-mtp3-$(date +%Y%m%d-%H%M%S).log"