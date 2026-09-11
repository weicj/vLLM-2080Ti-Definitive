"""Unit test verifying vLLM production integration."""
import pytest

def test_vllm_exl3_module_imports():
    """Verify that the EXL3 plugin imports."""
    try:
        from vllm_exl3 import exl3
    except ImportError as e:
        pytest.fail(f"Could not import vllm_exl3.exl3: {e}")

    assert hasattr(exl3, "Exl3Config")
