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

case "$VLLM_RUNTIME_BACKEND" in
  docker)
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
      --platform linux/amd64 --file docker/sm75/Dockerfile --target "$VLLM_DOCKER_VARIANT" \
      --build-arg "BASE_IMAGE=$docker_base" --build-arg "UPSTREAM_VERSION=$DOCKER_VLLM_VERSION" \
      --build-arg "FORK_RELEASE=$FORK_RELEASE" --build-arg "MAX_JOBS=${MAX_JOBS:-2}" \
      --tag "$VLLM_DOCKER_IMAGE" "$docker_context" || fail "Container build failed."
    printf '%s\n' "Build complete. Start with ./launcher.sh"
    exit 0
    ;;
  native) ;;
  *) fail "VLLM_RUNTIME_BACKEND must be docker or native." ;;
esac

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-13.0 /usr/local/cuda-13; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME="$candidate"
      export CUDA_HOME
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/bin" ]]; then
  case ":$PATH:" in
    *:"$CUDA_HOME/bin":*) ;;
    *) PATH="$CUDA_HOME/bin:$PATH"; export PATH ;;
  esac
fi
if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME:-}/.local/bin/uv" ]]; then
  PATH="${HOME}/.local/bin:$PATH"
  export PATH
fi

# Prefer prepared source trees when a previous build populated .deps. This
# avoids repeating large Git clones while keeping the upstream FetchContent
# defaults for a clean checkout.
if [[ -z "${TRITON_KERNELS_SRC_DIR:-}" ]]; then
  for candidate in \
    "$ROOT/.deps/triton_kernels-src/python/triton_kernels/triton_kernels" \
    "$ROOT/.deps/triton-kernels-tar"/triton-*/python/triton_kernels/triton_kernels; do
    if [[ -d "$candidate" ]]; then
      export TRITON_KERNELS_SRC_DIR="$candidate"
      break
    fi
  done
fi
if [[ -z "${VLLM_CUTLASS_SRC_DIR:-}" ]]; then
  for candidate in \
    "$ROOT/.deps/cutlass-src" \
    "$ROOT/.deps/cutlass-tar"/cutlass-*/; do
    if [[ -f "$candidate/include/cutlass/cutlass.h" ]]; then
      export VLLM_CUTLASS_SRC_DIR="$candidate"
      break
    fi
  done
fi

is_positive_integer() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

detect_cpu_threads() {
  local threads
  threads=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if ! is_positive_integer "$threads"; then
    threads=$(nproc 2>/dev/null || true)
  fi
  if ! is_positive_integer "$threads"; then
    threads=4
  fi
  echo "$threads"
}

detect_memory_gb() {
  local mem_kb mem_gb
  mem_kb=$(awk '/MemTotal:/ { print $2 }' /proc/meminfo 2>/dev/null || echo 0)
  mem_gb=$(( mem_kb / 1024 / 1024 ))
  if (( mem_gb < 1 )); then
    mem_gb=1
  fi
  echo "$mem_gb"
}

select_max_jobs() {
  local threads=$1
  local mem_gb=$2
  local cap=${3:-}
  local reserve_gb=3
  local per_job_gb=3
  local mem_limited_jobs

  mem_limited_jobs=$(( (mem_gb - reserve_gb) / per_job_gb ))
  (( mem_limited_jobs >= 1 )) || mem_limited_jobs=1
  if [[ -n "$cap" ]]; then
    (( mem_limited_jobs <= cap )) || mem_limited_jobs=$cap
  fi
  (( mem_limited_jobs <= threads )) || mem_limited_jobs=$threads

  echo "$mem_limited_jobs"
}

validate_max_jobs_range() {
  local jobs=$1
  local threads=$2
  is_positive_integer "$jobs" || fail "MAX_JOBS must be a positive integer."
  if (( jobs < 1 || jobs > threads )); then
    fail "MAX_JOBS must be between 1 and CPU_THREADS ($threads)."
  fi
}

configure_build_parallelism() {
  local answer
  [[ "$max_jobs_source" == auto* ]] || return 0
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    return 0
  fi

  cat <<EOF

Build thread selection:
  CPU threads detected: $cpu_threads
  Recommended build threads: $max_jobs
  Allowed range: 1-$cpu_threads

Press Enter to use the recommended value, or type a build thread count:
EOF
  while true; do
    read -r answer
    if [[ -z "$answer" ]]; then
      return 0
    fi
    if is_positive_integer "$answer" && (( answer <= cpu_threads )); then
      max_jobs=$answer
      max_jobs_source=manual-prompt
      return 0
    fi
    echo "Please enter a number from 1 to $cpu_threads, or press Enter for $max_jobs:"
  done
}

prompt_yes_no_timeout() {
  local prompt=$1 timeout_seconds=${2:-10} default_answer=${3:-y} answer=""
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    printf '%s\n' "$default_answer"
    return 0
  fi
  printf '%s ' "$prompt" >&2
  if read -r -t "$timeout_seconds" answer; then
    case "$answer" in
      y|Y|yes|YES) printf 'y\n' ;;
      n|N|no|NO) printf 'n\n' ;;
      *) printf '%s\n' "$default_answer" ;;
    esac
  else
    printf '%s\n' "$default_answer"
  fi
}

confirm_mirror_route() {
  local answer
  [[ -n "$git_mirror_prefix" || "$UV_INDEX_URL" != "$BUILD_PYPI_OFFICIAL_INDEX" ]] || return 0
  answer=$(prompt_yes_no_timeout \
    "Preflight selected third-party download mirrors. Continue? [Y/n]:" 10 y)
  [[ "$answer" != "n" ]] || fail "Build cancelled by user."
}

confirm_install() {
  local answer
  if [[ "${ASSUME_YES:-0}" == "1" || "${YES:-0}" == "1" || ! -t 0 ]]; then
    return 0
  fi
  cat <<EOF

This script will build and install vLLM 2080 Ti Definitive Edition into:
  $venv_dir

Build configuration:
  CUDA: $PRIMARY_CUDA_VERSION ($CUDA_HOME)
  Python: $python_version
  Build threads: $max_jobs ($max_jobs_source)
  PyPI index: $UV_INDEX_URL
  Git route: ${git_mirror_prefix:-https://github.com/}

It will download Python/CUDA dependencies, compile CUDA extensions, and write
logs under:
  $LOG_DIR

Continue? [y/N]:
EOF
  while true; do
    read -r answer
    case "$answer" in
      y|Y) return 0 ;;
      n|N|"") echo "Build cancelled."; exit 0 ;;
      *) echo "Please type y to continue or n to exit:" ;;
    esac
  done
}

cuda_backend="cu$(printf '%s' "$PRIMARY_CUDA_VERSION" | cut -d. -f1,2 | tr -d '.')"
cpu_threads=${CPU_THREADS:-$(detect_cpu_threads)}
memory_gb=${MEMORY_GB:-$(detect_memory_gb)}
auto_max_jobs_cap=${BUILD_AUTO_MAX_JOBS_CAP:-8}
is_positive_integer "$cpu_threads" || fail "CPU_THREADS must be a positive integer when set explicitly."
is_positive_integer "$memory_gb" || fail "MEMORY_GB must be a positive integer when set explicitly."
is_positive_integer "$auto_max_jobs_cap" || fail "BUILD_AUTO_MAX_JOBS_CAP must be a positive integer when set explicitly."

if [[ -n "${BUILD_MAX_JOBS:-}" && -n "${MAX_JOBS:-}" && "${BUILD_MAX_JOBS}" != "${MAX_JOBS}" ]]; then
  fail "BUILD_MAX_JOBS and MAX_JOBS must match when both are set."
fi
if [[ -n "${BUILD_MAX_JOBS:-}" ]]; then
  max_jobs=$BUILD_MAX_JOBS
  max_jobs_source=env:BUILD_MAX_JOBS
elif [[ -n "${MAX_JOBS:-}" ]]; then
  max_jobs=$MAX_JOBS
  max_jobs_source=env:MAX_JOBS
else
  auto_jobs_without_cap=$(select_max_jobs "$cpu_threads" "$memory_gb" "$cpu_threads")
  max_jobs=$(select_max_jobs "$cpu_threads" "$memory_gb" "$auto_max_jobs_cap")
  if (( max_jobs < auto_jobs_without_cap )); then
    max_jobs_source=auto-cap
  elif (( max_jobs < cpu_threads )); then
    max_jobs_source=auto-memory
  else
    max_jobs_source=auto
  fi
fi
allow_host_mismatch=${ALLOW_HOST_MISMATCH:-0}
require_primary_env=${REQUIRE_PRIMARY_ENV:-1}
python_version=${PYTHON_VERSION:-$PRIMARY_PYTHON_VERSION}
venv_dir=${VENV_DIR:-"$ROOT/.venv"}
python_bin="$venv_dir/bin/python"
configured_git_mirror_prefix=${BUILD_GIT_MIRROR_PREFIX:-}
git_mirror_prefix=$configured_git_mirror_prefix
flashqla_repo=${FLASHQLA_REPO:-https://github.com/weicj/FlashQLA-SM70-SM75.git}
flashqla_dir=${FLASHQLA_DIR:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
flashqla_enabled=${FLASHQLA_ENABLED:-1}
flashqla_clone_timeout=${FLASHQLA_CLONE_TIMEOUT:-180}
skip_vllm_build=${SKIP_VLLM_BUILD:-0}

BUILD_PYPI_OFFICIAL_INDEX=${BUILD_PYPI_OFFICIAL_INDEX:-https://pypi.org/simple}
BUILD_PYPI_FOREIGN_INDEX=${BUILD_PYPI_FOREIGN_INDEX:-https://pypi.python.org/simple}
BUILD_PYPI_DOMESTIC_INDEX=${BUILD_PYPI_DOMESTIC_INDEX:-https://mirrors.aliyun.com/pypi/simple}
BUILD_TORCH_DOMESTIC_INDEX=${BUILD_TORCH_DOMESTIC_INDEX:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu130/}
BUILD_TORCH_INDEX=${BUILD_TORCH_INDEX:-}
BUILD_GIT_FOREIGN_REPO_PREFIX=${BUILD_GIT_FOREIGN_REPO_PREFIX:-https://gh-proxy.com/}
BUILD_GIT_DOMESTIC_REPO_PREFIX=${BUILD_GIT_DOMESTIC_REPO_PREFIX:-https://ghfast.top/}
BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS=${BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS:-5}
BUILD_PREFLIGHT_TRANSFER_SECONDS=${BUILD_PREFLIGHT_TRANSFER_SECONDS:-5}

validate_max_jobs_range "$max_jobs" "$cpu_threads"
is_positive_integer "$flashqla_clone_timeout" || fail "FLASHQLA_CLONE_TIMEOUT must be a positive integer."
is_positive_integer "$BUILD_PREFLIGHT_TRANSFER_SECONDS" || fail "BUILD_PREFLIGHT_TRANSFER_SECONDS must be a positive integer."
[[ "$skip_vllm_build" == "0" || "$skip_vllm_build" == "1" ]] ||
  fail "SKIP_VLLM_BUILD must be 0 or 1."
require_command uv

measure_network_url_ms() {
  local url=$1 timeout=${2:-5} elapsed
  if command -v curl >/dev/null 2>&1; then
    elapsed=$(curl -L --fail --silent --show-error \
      --connect-timeout "$timeout" --max-time "$timeout" \
      -o /dev/null -w '%{time_total}' "$url") || return 1
    awk -v seconds="$elapsed" 'BEGIN { printf "%d\n", seconds * 1000 }'
    return 0
  elif command -v wget >/dev/null 2>&1; then
    local start end
    start=$(python3 -c 'import time; print(time.monotonic_ns())')
    wget -q --timeout="$timeout" -O /dev/null "$url" || return 1
    end=$(python3 -c 'import time; print(time.monotonic_ns())')
    printf '%s\n' "$(((end - start) / 1000000))"
    return 0
  else
    return 1
  fi
}

pep440_version() {
  sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+[^-]*)-([0-9]+)-g([0-9a-f]+)$/\1+\2.g\3/' <<<"$1"
}

measure_network_url_bytes() {
  local url=$1 timeout=${2:-5} bytes
  command -v curl >/dev/null 2>&1 || {
    echo 0
    return 0
  }
  bytes=$(curl -L --silent --connect-timeout "$timeout" --max-time "$timeout" \
    -o /dev/null -w '%{size_download}' "$url" 2>/dev/null || true)
  [[ "$bytes" =~ ^[0-9]+$ ]] || bytes=0
  echo "$bytes"
}

run_build_network_preflight() {
  local -a modes=(official foreign domestic) pypi_urls=("$BUILD_PYPI_OFFICIAL_INDEX" "$BUILD_PYPI_FOREIGN_INDEX" "$BUILD_PYPI_DOMESTIC_INDEX")
  local -a prefixes=("" "$BUILD_GIT_FOREIGN_REPO_PREFIX" "$BUILD_GIT_DOMESTIC_REPO_PREFIX")
  local best_pypi_mode=official best_pypi_ms=999999999 best_git_mode=official best_git_bytes=-1
  local mode pypi_probe git_probe git_transfer_probe pypi_ms git_ms git_bytes i
  echo "Build preflight: benchmarking PyPI and Git routes..."
  for i in 0 1 2; do
    mode=${modes[$i]}; pypi_probe=${pypi_urls[$i]%/}/pip/
    git_probe="${prefixes[$i]}${flashqla_repo}/info/refs?service=git-upload-pack"
    git_transfer_probe="${prefixes[$i]}https://github.com/nvidia/cutlass/archive/refs/tags/v4.7.1.tar.gz"
    pypi_ms=$(measure_network_url_ms "$pypi_probe" "$BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS" || echo 999999)
    git_ms=$(measure_network_url_ms "$git_probe" "$BUILD_PREFLIGHT_SAMPLE_TIMEOUT_SECONDS" || echo 999999)
    git_bytes=$(measure_network_url_bytes "$git_transfer_probe" "$BUILD_PREFLIGHT_TRANSFER_SECONDS")
    echo "Build preflight: $mode PyPI=${pypi_ms}ms Git=${git_ms}ms Git-transfer=${git_bytes}B/${BUILD_PREFLIGHT_TRANSFER_SECONDS}s"
    if (( pypi_ms < best_pypi_ms )); then best_pypi_ms=$pypi_ms; best_pypi_mode=$mode; fi
    if (( git_bytes > best_git_bytes )); then best_git_bytes=$git_bytes; best_git_mode=$mode; fi
  done
  case "$best_pypi_mode" in
    official) export UV_INDEX_URL="$BUILD_PYPI_OFFICIAL_INDEX" ;;
    foreign) export UV_INDEX_URL="$BUILD_PYPI_FOREIGN_INDEX" ;;
    domestic)
      export UV_INDEX_URL="$BUILD_PYPI_DOMESTIC_INDEX"
      BUILD_TORCH_INDEX=${BUILD_TORCH_INDEX:-$BUILD_TORCH_DOMESTIC_INDEX}
      ;;
  esac
  if [[ -n "$configured_git_mirror_prefix" ]]; then
    git_mirror_prefix=$configured_git_mirror_prefix
  else
    case "$best_git_mode" in
      official) git_mirror_prefix="" ;;
      foreign) git_mirror_prefix="$BUILD_GIT_FOREIGN_REPO_PREFIX" ;;
      domestic) git_mirror_prefix="$BUILD_GIT_DOMESTIC_REPO_PREFIX" ;;
    esac
  fi
  export BUILD_GIT_MIRROR_PREFIX="$git_mirror_prefix"
  if [[ -n "$git_mirror_prefix" ]]; then
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.${git_mirror_prefix}https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="https://github.com/"
  fi
  echo "Build preflight: selected $best_pypi_mode package route (UV_INDEX_URL=$UV_INDEX_URL)"
  echo "Build preflight: Git route=${git_mirror_prefix:-https://github.com/}"
  confirm_mirror_route
}

install_torch_from_mirror() {
  [[ -n "$BUILD_TORCH_INDEX" ]] || return 0
  echo "Installing Torch/Triton from mirror: $BUILD_TORCH_INDEX"
  uv pip install --python "$python_bin" \
    --index-url "$BUILD_TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    "torch==${PRIMARY_TORCH_VERSION}+${cuda_backend}" \
    "triton==3.7.1"
}

install_vllm_build_tools() {
  echo "Installing vLLM build tools from $UV_INDEX_URL"
  uv pip install --python "$python_bin" \
    "cmake>=3.26.1" ninja "packaging>=24.2" \
    "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
    "setuptools-rust>=1.9.0" wheel "jinja2>=3.1.6"
}

if [[ ! -f pyproject.toml || ! -d vllm ]]; then
  fail "Run this script from the 0.2.2-post3 source tree."
fi

check_primary_host() {
  local failures=()
  local nvcc_major gcc_major kernel_major

  if command -v nvcc >/dev/null 2>&1; then
    nvcc_major=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)
    if [[ "$nvcc_major" != "${PRIMARY_CUDA_VERSION%%.*}" ]]; then
      failures+=("nvcc major ${nvcc_major:-unknown} (expected ${PRIMARY_CUDA_VERSION%%.*})")
    fi
  else
    failures+=("nvcc is not on PATH")
  fi

  if command -v gcc >/dev/null 2>&1; then
    gcc_major=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$gcc_major" != "$PRIMARY_GCC_MAJOR" ]]; then
      failures+=("GCC major ${gcc_major:-unknown} (expected $PRIMARY_GCC_MAJOR)")
    fi
  else
    failures+=("gcc is not on PATH")
  fi

  kernel_major=$(uname -r | cut -d. -f1)
  if ! is_positive_integer "$kernel_major" || (( kernel_major < PRIMARY_MIN_KERNEL_MAJOR )); then
    failures+=("Linux kernel ${kernel_major:-unknown} (expected >= $PRIMARY_MIN_KERNEL_MAJOR)")
  fi

  if ((${#failures[@]} > 0)); then
    printf 'Primary host checks: %s\n' "${failures[*]}"
    if [[ "$require_primary_env" == "1" && "$allow_host_mismatch" != "1" ]]; then
      fail "Host does not match the 0.2.2-post3 target. Set ALLOW_HOST_MISMATCH=1 only for a non-target dry run."
    fi
    echo "Continuing because ALLOW_HOST_MISMATCH=1 or REQUIRE_PRIMARY_ENV=0."
  else
    echo "Primary host checks: CUDA ${PRIMARY_CUDA_VERSION}, GCC ${PRIMARY_GCC_MAJOR}, kernel >= ${PRIMARY_MIN_KERNEL_MAJOR}"
  fi
}

check_cuda_glibc_compatibility() {
  local header="$CUDA_HOME/targets/x86_64-linux/include/crt/math_functions.h"
  local glibc_version

  glibc_version=$(ldd --version 2>/dev/null | awk 'NR == 1 { print $NF }')
  [[ "$glibc_version" =~ ^[0-9]+\.[0-9]+$ ]] || return 0
  if [[ "$(printf '%s\n%s\n' 2.41 "$glibc_version" | sort -V | head -n 1)" != "2.41" ]]; then
    return 0
  fi

  if ! grep -q '__GLIBC_PREREQ(2,41)' "$header" || \
    ! grep -q '_NV_RSQRT_SPECIFIER' "$header"; then
    fail "CUDA $PRIMARY_CUDA_VERSION requires toolchain-patches/cuda-13.0-glibc-2.41-rsqrt.patch on glibc $glibc_version. Apply it to CUDA_HOME before rebuilding."
  fi
}

prepare_flashqla_sm75() {
  [[ "$flashqla_enabled" == "1" ]] || {
    echo "FlashQLA SM70/SM75 backend: disabled (FLASHQLA_ENABLED=$flashqla_enabled)"
    return 0
  }

  mkdir -p "$(dirname -- "$flashqla_dir")"
  if [[ ! -e "$flashqla_dir" ]]; then
    echo "Fetching FlashQLA SM70/SM75 backend from $flashqla_repo"
    clone_flashqla() {
      local source=$1
      if command -v timeout >/dev/null 2>&1; then
        timeout --foreground "$flashqla_clone_timeout" \
          git clone --depth=1 "$source" "$flashqla_dir"
      else
        git clone --depth=1 "$source" "$flashqla_dir"
      fi
    }
    if [[ -n "$git_mirror_prefix" ]]; then
      clone_flashqla "${git_mirror_prefix}${flashqla_repo#https://github.com/}" || {
        rm -rf "$flashqla_dir"
        clone_flashqla "$flashqla_repo"
      }
    else
      clone_flashqla "$flashqla_repo"
    fi
  fi

  [[ -f "$flashqla_dir/flash_qla/ops/gated_delta_rule/legacy/sm_legacy.py" ]] ||
    fail "FlashQLA checkout is missing the SM70/SM75 legacy backend"

  "$python_bin" tools/patch_flashqla_sm75_imports.py \
    "$flashqla_dir" "$ROOT/tools/flashqla_sm75_patches"

  echo "Installing FlashQLA Python package into $venv_dir"
  uv pip install --python "$python_bin" --no-deps -e "$flashqla_dir"

  local extension_dir="${TORCH_EXTENSIONS_DIR:-$flashqla_dir/.torch_extensions_vllm_flashqla_legacy}"
  echo "Building FlashQLA SM75 legacy extension in $extension_dir"
  CUDA_HOME="$CUDA_HOME" \
  CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}" \
  CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  TORCH_EXTENSIONS_DIR="$extension_dir" \
  PYTHONPATH="$flashqla_dir${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" - <<'PY'
from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import _load_ext

extension = _load_ext()
required = ("gdn_forward", "gdn_forward_varlen")
missing = [name for name in required if not hasattr(extension, name)]
if missing:
    raise SystemExit(
        "FlashQLA extension is stale; missing exported symbols: "
        + ", ".join(missing)
        + ". Re-run the build after applying the SM75 packed-varlen patch."
    )
print(f"FlashQLA extension: {extension.__file__}")
print("FlashQLA symbols: " + ", ".join(required))
PY

  export FLASHQLA_DIR="$flashqla_dir"
  export TORCH_EXTENSIONS_DIR="$extension_dir"
  export PYTHONPATH="$flashqla_dir${PYTHONPATH:+:$PYTHONPATH}"
}

configure_build_parallelism

echo "============================================================"
echo "vLLM 2080 Ti Definitive Edition ${FORK_RELEASE} source build"
echo "Upstream base: vLLM ${BASE_VLLM_VERSION}"
echo "Target: CUDA ${PRIMARY_CUDA_VERSION} / Torch ${PRIMARY_TORCH_VERSION} / SM75"
echo "Build parallelism: ${max_jobs} (${max_jobs_source}; CPU=${cpu_threads}, memory=${memory_gb}GiB, auto cap=${auto_max_jobs_cap})"
echo "Build log: $LOG_FILE"
echo "============================================================"

check_primary_host
check_cuda_glibc_compatibility
run_build_network_preflight
confirm_install

export MAX_JOBS="$max_jobs"
export BUILD_MAX_JOBS=${BUILD_MAX_JOBS:-$MAX_JOBS}
export CPU_THREADS="$cpu_threads"
export MEMORY_GB="$memory_gb"
export AUTO_MAX_JOBS_CAP="$auto_max_jobs_cap"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-$cuda_backend}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

if [[ -n "$git_mirror_prefix" ]]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${git_mirror_prefix}https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
fi

# A source snapshot made with git archive, or a copied worktree whose `.git`
# file points at unavailable metadata, has no usable SCM information. Keep
# the package version deterministic in both supported validation layouts.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  snapshot_version=$(pep440_version "$BASE_VLLM_VERSION")
  export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-$snapshot_version}"
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM:-$snapshot_version}"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Creating Python ${python_version} environment at $venv_dir"
  uv venv --python "$python_version" "$venv_dir"
fi

[[ -x "$python_bin" ]] || fail "Python environment was not created: $python_bin"

# Build helpers such as ninja are installed into the venv by uv. Keep the
# editable FlashQLA build and the vLLM build on the same toolchain PATH.
case ":$PATH:" in
  *:"$venv_dir/bin":*) ;;
  *) PATH="$venv_dir/bin:$PATH"; export PATH ;;
esac

if [[ "$skip_vllm_build" == "1" ]]; then
  echo "Skipping vLLM build (SKIP_VLLM_BUILD=1); validating existing editable install"
else
  install_torch_from_mirror
  install_vllm_build_tools
  echo "Installing and compiling vLLM with uv torch backend $cuda_backend"
  uv pip install --python "$python_bin" --no-build-isolation -e "." "--torch-backend=$cuda_backend"
fi

"$python_bin" tools/patch_torch_inductor_e8m0.py
"$python_bin" tools/check_torch_inductor_e8m0.py

prepare_flashqla_sm75

echo "Checking the resulting runtime"
EXPECTED_TORCH_VERSION="$PRIMARY_TORCH_VERSION" \
EXPECTED_CUDA_BACKEND="$cuda_backend" \
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
FLASHQLA_ENABLED="$flashqla_enabled" \
  "$python_bin" - <<'PY'
import os

import torch
import vllm

expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
expected_backend = os.environ["EXPECTED_CUDA_BACKEND"]
if not torch.__version__.startswith(expected_torch):
    raise SystemExit(f"unexpected torch version: {torch.__version__}")
if torch.version.cuda is None:
    raise SystemExit("the installed torch build has no CUDA runtime")

print(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")
print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA runtime: {torch.version.cuda} (requested {expected_backend})")
print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: capability={torch.cuda.get_device_capability(index)}")
if os.environ.get("FLASHQLA_ENABLED") == "1":
    from flash_qla.ops.gated_delta_rule.legacy.sm_legacy import _load_ext

    extension = _load_ext()
    required = ("gdn_forward", "gdn_forward_varlen")
    missing = [name for name in required if not hasattr(extension, name)]
    if missing:
        raise SystemExit(
            "FlashQLA extension is stale; missing exported symbols: "
            + ", ".join(missing)
        )
    print(f"FlashQLA extension: {extension.__file__}")
    print("FlashQLA symbols: " + ", ".join(required))
PY

echo "BUILD SUCCEEDED"
echo "Runtime: $python_bin"
