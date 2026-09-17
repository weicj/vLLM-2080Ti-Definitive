#!/usr/bin/env bash
# Stop hook for the vllm-2080ti systemd user service.
# The launcher daemonizes the server (nohup setsid), so systemd cannot track
# it directly. This script mirrors the launcher's own stop logic (pid files +
# process-tree kill + orphan scan) to make sure everything is really gone.
set -uo pipefail

LOG_DIR=/home/john/vLLM-2080Ti-Definitive/run-logs

stop_tree() {
  local pid=$1 sig=$2 child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    stop_tree "$child" "$sig"
  done
  kill "-$sig" "$pid" 2>/dev/null || true
}

stop_pid_file() {
  local pid_file=$1 pid pgid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
    rm -f "$pid_file"
    return 0
  fi

  # The server was started with setsid, so it is its own process group:
  # killing the group takes the engine + workers with it.
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -n "$pgid" && "$pgid" == "$pid" ]]; then
    kill -- "-$pid" 2>/dev/null || true
  fi
  stop_tree "$pid" TERM

  local i
  for i in $(seq 1 40); do
    [[ -d "/proc/$pid" ]] || break
    sleep 0.5
  done

  if [[ -d "/proc/$pid" ]]; then
    [[ -n "$pgid" && "$pgid" == "$pid" ]] && kill -9 -- "-$pid" 2>/dev/null || true
    stop_tree "$pid" KILL
  fi
  rm -f "$pid_file"
}

# 1) Stop every service recorded by the launcher (pid files)
shopt -s nullglob
for pid_file in "$LOG_DIR"/vllm-*.pid; do
  stop_pid_file "$pid_file" || true
done

# 2) Kill any remaining vLLM processes (orphans, VLLM:: workers)
pids=$(pgrep -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true)
if [[ -n "$pids" ]]; then
  kill $pids 2>/dev/null || true
  sleep 3
  for pid in $pids; do
    [[ -d "/proc/$pid" ]] && kill -9 "$pid" 2>/dev/null || true
  done
fi
while read -r pid; do
  [[ -n "$pid" ]] || continue
  kill "$pid" 2>/dev/null || true
done < <(ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^VLLM::/ {print $1}')
sleep 1
while read -r pid; do
  [[ -n "$pid" ]] || continue
  [[ -d "/proc/$pid" ]] && kill -9 "$pid" 2>/dev/null || true
done < <(ps -eo pid=,comm= 2>/dev/null | awk '$2 ~ /^VLLM::/ {print $1}')

# 3) Remove stale pid files
rm -f "$LOG_DIR"/vllm-*.pid

exit 0
