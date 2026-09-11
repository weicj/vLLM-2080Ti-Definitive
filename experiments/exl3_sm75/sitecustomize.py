"""Opt-in plugin registration for isolated EXL3 worker-process experiments."""

import os


if os.environ.get("VLLM_EXL3_EXPERIMENT_BOOTSTRAP") == "1":
    from vllm_exl3 import register

    register()
