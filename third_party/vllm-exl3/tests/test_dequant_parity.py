"""CPU parity tests for the EXL3 IEEE-754 and procedural codebook reference."""

import math

import pytest

torch = pytest.importorskip("torch")

from vllm_exl3.dequant import (
    decode_codebook_bits_py,
    decode_mcg_py,
    decode_mul1_py,
    float_to_half_bits_py,
    half_bits_to_float_py,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2**-15, 0x0200),
        (1.00075, 0x3C01),
        (-1.00075, 0xBC01),
        (2**-24, 0x0001),
        (2**-25, 0x0000),
        (1.0 + 2**-11, 0x3C00),
        (1.0 + 3 * 2**-11, 0x3C02),
        (65504.0, 0x7BFF),
        (65520.0, 0x7C00),
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (math.inf, 0x7C00),
        (-math.inf, 0xFC00),
    ],
)
def test_float_to_half_bits_edge_vectors(value: float, expected: int) -> None:
    assert float_to_half_bits_py(value) == expected


def test_nan_conversion_and_half_decoding() -> None:
    nan_bits = float_to_half_bits_py(float("nan"))
    assert (nan_bits & 0x7C00) == 0x7C00
    assert nan_bits & 0x03FF
    assert math.isnan(half_bits_to_float_py(nan_bits))
    assert math.copysign(1.0, half_bits_to_float_py(0x8000)) < 0


@pytest.mark.parametrize(
    "value",
    [
        -70000.0,
        -65504.0,
        -32.125,
        -2**-14,
        -2**-15,
        -2**-24,
        -0.0,
        0.0,
        2**-24,
        2**-15,
        2**-14,
        0.333251953125,
        1.00075,
        32.125,
        65504.0,
        70000.0,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_float_to_half_bits_matches_torch(value: float) -> None:
    expected = int(
        torch.tensor(value, dtype=torch.float32).half().view(torch.int16).item()
    ) & 0xFFFF
    actual = float_to_half_bits_py(value)
    if math.isnan(value):
        assert (actual & 0x7C00) == 0x7C00 and actual & 0x03FF
        assert (expected & 0x7C00) == 0x7C00 and expected & 0x03FF
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("window", "mcg_bits", "mul1_bits", "mcg_value", "mul1_value"),
    [
        (0x0000, 0x3F60, 0xC2E8, 1.84375, -3.453125),
        (0x0001, 0x304E, 0x3921, 0.134521484375, 0.6410751342773438),
        (0x1234, 0x3E0F, 0xB89E, 1.5142822265625, -0.5770339965820312),
        (0x7FFF, 0x4172, 0x3FCA, 2.72216796875, 1.9471588134765625),
        (0xFFFF, 0x3ACD, 0xB936, 0.85009765625, -0.6514739990234375),
    ],
)
def test_procedural_codebook_vectors(
    window: int,
    mcg_bits: int,
    mul1_bits: int,
    mcg_value: float,
    mul1_value: float,
) -> None:
    assert decode_codebook_bits_py(window, 1) == mcg_bits
    assert decode_codebook_bits_py(window, 2) == mul1_bits
    assert decode_mcg_py(window) == mcg_value
    assert decode_mul1_py(window) == mul1_value
