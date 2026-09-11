"""Deprecated alias for :mod:`vllm_exl3`; will be removed in a future release.

Import ``vllm_exl3`` instead. This shim keeps older recipes and wheels that
import ``glm53_exl3_plugin`` working across the package rename. Imports stay
lazy so the vLLM import chain is only touched when ``register()`` runs.
"""


def register() -> None:
    from vllm_exl3 import register as _register

    _register()


def __getattr__(name):
    if name == "exl3":
        from vllm_exl3 import exl3 as _exl3

        return _exl3
    raise AttributeError(name)
