#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

PROJECT_RELEASE_FILE=${PROJECT_RELEASE_FILE:-"$ROOT/PROJECT_RELEASE.env"}
# shellcheck source=/dev/null
source "$PROJECT_RELEASE_FILE"

LOG_DIR=${BUILD_LOG_DIR:-"$ROOT/build-logs"}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/build-$STAMP.log"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() {
  echo "BUILD FAILED: $*" >&2
  echo "Log: $LOG_FILE" >&2
  exit 1
}

# shellcheck source=tools/docker_runtime.sh
source "$ROOT/tools/docker_runtime.sh"

command -v "$VLLM_DOCKER_COMMAND" >/dev/null || fail "Docker is required for the container build."
"$VLLM_DOCKER_COMMAND" info >/dev/null || fail "Docker daemon is unavailable."
docker_base=${VLLM_DOCKER_BASE_IMAGE:-$DOCKER_BASE_IMAGE}
printf '%s\n' "$PROJECT_NAME v$FORK_RELEASE" "Official base: $docker_base" "Runtime image: $VLLM_DOCKER_IMAGE"
docker_context=$(mktemp -d)
trap 'rm -rf -- "$docker_context"' EXIT
mkdir -p "$docker_context/docker" "$docker_context/tools"
cp -a docker/sm75 "$docker_context/docker/"
cp -a tools/flashqla_sm75_patches "$docker_context/tools/"
cp LICENSE "$docker_context/"
"$VLLM_DOCKER_COMMAND" build \
  --platform linux/amd64 --file docker/sm75/Dockerfile \
  --build-arg "BASE_IMAGE=$docker_base" --build-arg "UPSTREAM_VERSION=$DOCKER_VLLM_VERSION" \
  --build-arg "FORK_RELEASE=$FORK_RELEASE" --build-arg "MAX_JOBS=${MAX_JOBS:-2}" \
  --tag "$VLLM_DOCKER_IMAGE" "$docker_context" || fail "Container build failed."
printf '%s\n' "Build complete. Start with ./launcher.sh"
