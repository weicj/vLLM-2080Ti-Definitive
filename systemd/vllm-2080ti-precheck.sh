#!/usr/bin/env bash
# ExecStartPre for vllm-2080ti.service.
# On boot the NVIDIA driver can take a moment to expose the GPUs. Wait until
# nvidia-smi sees both cards before the launcher tries to load the model.
set -u

want=${EXPECTED_GPU_COUNT:-2}
for i in $(seq 1 60); do
  count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  if [[ "$count" -ge "$want" ]]; then
    exit 0
  fi
  sleep 2
done

echo "vllm-2080ti: only ${count:-0}/${want} GPUs visible after 120s" >&2
exit 1
