"""Standalone CPU reference for EXL3 half and procedural codebook decoding."""

from __future__ import annotations

import math
import struct

MCG_MULTIPLIER = 0xCBAC1FED
MUL1_MULTIPLIER = 0x83DCD12D


def _float32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except OverflowError:
        sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
        return sign | 0x7F800000


def float_to_half_bits_py(value: float) -> int:
    """Convert a Python number as float32 to IEEE-754 binary16 with RTNE."""
    raw = _float32_bits(value)
    sign = (raw >> 16) & 0x8000
    exp_bits = (raw >> 23) & 0xFF
    mant = raw & 0x7FFFFF
    if exp_bits == 0xFF:
        if mant == 0:
            return sign | 0x7C00
        payload = (mant >> 13) | 0x200
        return sign | 0x7C00 | (payload & 0x3FF)

    exp = exp_bits - 127
    significand = mant | 0x800000
    if exp < -14:
        shift = -exp - 1
        if shift >= 32:
            return sign
        rounded = significand >> shift
        remainder = significand & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and rounded & 1):
            rounded += 1
        return sign | (rounded & 0x7FF)

    rounded_mant = mant >> 13
    remainder = mant & 0x1FFF
    if remainder > 0x1000 or (remainder == 0x1000 and rounded_mant & 1):
        rounded_mant += 1
    half_exp = exp + 15
    if rounded_mant == 0x400:
        rounded_mant = 0
        half_exp += 1
    if half_exp >= 31:
        return sign | 0x7C00
    return sign | (half_exp << 10) | rounded_mant


def half_bits_to_float_py(bits: int) -> float:
    """Decode an unsigned IEEE-754 binary16 bit pattern to Python float."""
    bits &= 0xFFFF
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        value = math.ldexp(float(mant), -24)
        return -value if sign else value
    if exp == 0x1F:
        return float("nan") if mant else (-math.inf if sign else math.inf)
    value = math.ldexp(1.0 + mant / 1024.0, exp - 15)
    return -value if sign else value


def _lop3_6a(a: int, b: int, c: int) -> int:
    out = 0
    for bit in range(32):
        index = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        if (0x6A >> index) & 1:
            out |= 1 << bit
    return out


def _half_add_bits(a: int, b: int) -> int:
    return float_to_half_bits_py(half_bits_to_float_py(a) + half_bits_to_float_py(b))


def _half_mul_bits(a: int, b: int) -> int:
    return float_to_half_bits_py(half_bits_to_float_py(a) * half_bits_to_float_py(b))


def _half_fma_bits(a: int, b: int, c: int) -> int:
    return float_to_half_bits_py(
        half_bits_to_float_py(a) * half_bits_to_float_py(b)
        + half_bits_to_float_py(c)
    )


def decode_codebook_bits_py(window: int, cb: int) -> int:
    """Return the half bit pattern for EXL3 procedural codebook ``cb``."""
    x = int(window) & 0xFFFF
    if cb == 0:
        x = (x * 89226354 + 64248484) & 0xFFFFFFFF
        x = _lop3_6a(x, 0x8FFF8FFF, 0x3B603B60)
        return _half_add_bits(x, x >> 16)
    if cb == 1:
        x = (x * MCG_MULTIPLIER) & 0xFFFFFFFF
        x = _lop3_6a(x, 0x8FFF8FFF, 0x3B603B60)
        return _half_add_bits(x, x >> 16)
    x = (x * MUL1_MULTIPLIER) & 0xFFFFFFFF
    total = 0x6400 + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    return _half_fma_bits(total, 0x1EEE, 0xC931)


def decode_mcg_py(window: int) -> float:
    """Decode one MCG window using the C++ reference's final float sum."""
    x = ((int(window) & 0xFFFF) * MCG_MULTIPLIER) & 0xFFFFFFFF
    x = _lop3_6a(x, 0x8FFF8FFF, 0x3B603B60)
    return half_bits_to_float_py(x) + half_bits_to_float_py(x >> 16)


def decode_mul1_py(window: int) -> float:
    """Decode one MUL1 window using the C++ reference's float expression."""
    x = ((int(window) & 0xFFFF) * MUL1_MULTIPLIER) & 0xFFFFFFFF
    total = 0x6400 + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    return half_bits_to_float_py(total) * half_bits_to_float_py(0x1EEE) + half_bits_to_float_py(0xC931)
