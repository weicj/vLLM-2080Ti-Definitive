"""Register the routed-expert EXL3 implementation with a local vLLM runtime."""

__all__ = [
    "register",
    "get_speculative_draft_tokens",
    "parse_speculative_schedule",
    "is_adaptive_verification_enabled",
    "filter_speculative_candidates",
    "compute_mla_kv_cache_bytes",
    "validate_context_scaling",
]


def register() -> None:
    # Importing the module executes its register_quantization_config decorator.
    from . import exl3 as _exl3  # noqa: F401


def __getattr__(name: str):
    """Expose scheduler helpers without making package import eager.

    vLLM loads this package as an entry point during startup.  Keeping the
    implementation import lazy preserves the existing lightweight entry-point
    behaviour while still supporting ``from vllm_exl3 import ...``.
    """
    if name in {
        "get_speculative_draft_tokens",
        "parse_speculative_schedule",
        "is_adaptive_verification_enabled",
        "filter_speculative_candidates",
        "compute_mla_kv_cache_bytes",
        "validate_context_scaling",
    }:
        from . import exl3

        return getattr(exl3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
