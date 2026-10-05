#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Container-only build of vLLM 2080 Ti Definitive Edition (github.com/weicj).
set -euo pipefail

cd /src
# Use the same version contract as the host build, without its host preflight.
source PROJECT_RELEASE.env
export VLLM_VERSION_OVERRIDE
VLLM_VERSION_OVERRIDE=$(python - <<'PY'
import tomllib
from pathlib import Path

config = tomllib.loads(Path("pyproject.toml").read_text())
print(config["tool"]["setuptools_scm"]["fallback_version"])
PY
)
export SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION_OVERRIDE"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="$VLLM_VERSION_OVERRIDE"
export VLLM_RS_BUILD_VERSION="$VLLM_VERSION_OVERRIDE"
export UV_TORCH_BACKEND="cu${PRIMARY_CUDA_VERSION//./}"
export EXPECTED_TORCH_VERSION="$VALIDATED_TORCH_VERSION"
export EXPECTED_PYTHON_VERSION="$PRIMARY_PYTHON_VERSION"
export EXPECTED_CUDA_VERSION="$PRIMARY_CUDA_VERSION"

check_toolchain() {
    python - <<'PY'
import os
import platform
import subprocess

import torch

expected = os.environ
assert platform.python_version().startswith(expected["EXPECTED_PYTHON_VERSION"] + ".")
assert torch.__version__ == expected["EXPECTED_TORCH_VERSION"], torch.__version__
assert torch.version.cuda == expected["EXPECTED_CUDA_VERSION"], torch.version.cuda
nvcc = subprocess.check_output([os.environ["CUDACXX"], "--version"], text=True)
assert f"release {torch.version.cuda}," in nvcc, nvcc
assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.5"
print(f"Build toolchain: Python {platform.python_version()}, Torch {torch.__version__}")
PY
}

case "${1:-}" in
    dependencies)
        uv pip install --python "$VIRTUAL_ENV/bin/python" \
            -r requirements/build/cuda.txt -r requirements/build/rust.txt
        check_toolchain
        # Ask the normal packaging backend for its dependency metadata. This
        # preserves setup.py's CUDA-specific exclusions without duplicating them.
        python - <<'PY'
from email.parser import Parser
from pathlib import Path
from setuptools.build_meta import prepare_metadata_for_build_wheel

out = Path("/tmp/vllm-metadata")
out.mkdir(exist_ok=True)
name = prepare_metadata_for_build_wheel(str(out))
metadata = Parser().parsestr((out / name / "METADATA").read_text())
Path("/tmp/vllm-runtime-requirements.txt").write_text(
    "\n".join(metadata.get_all("Requires-Dist", [])) + "\n"
)
PY
        uv pip install --python "$VIRTUAL_ENV/bin/python" \
            -r /tmp/vllm-runtime-requirements.txt
        uv pip check
        check_toolchain
        ;;
    wheel)
        check_toolchain
        # Build Rust from this checkout. The existing helper installs the
        # rust-toolchain.toml version and fails if either artifact cannot build.
        bash build_rust.sh
        python -m build --wheel --no-isolation --outdir /tmp/vllm-wheels
        uv pip install --python "$VIRTUAL_ENV/bin/python" /tmp/vllm-wheels/*.whl
        python tools/patch_torch_inductor_e8m0.py
        python tools/check_torch_inductor_e8m0.py
        uv pip check
        # No FlashQLA _load_ext(): Docker builds have no GPU. Its patched source
        # and JIT toolchain are included in the final stage instead.
        printf '%s\n' 'SM75 wheel built; GPU kernels and FlashQLA JIT require runtime validation.'
        ;;
    *)
        printf 'Usage: %s {dependencies|wheel}\n' "$0" >&2
        exit 2
        ;;
esac
