"""CPU regression tests for EXL3 SwiGLU activation and dispatch semantics."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class _FixedProjection:
    def __init__(self, value: float) -> None:
        self.value = value

    def forward(self, x: torch.Tensor, _params: dict, *, out_dtype):
        return torch.full((x.shape[0], 1), self.value, dtype=out_dtype)


class _IdentityProjection:
    def forward(self, x: torch.Tensor, _params: dict, *, out_dtype):
        return x.to(dtype=out_dtype)


def _run_activation(gate: float, up: float, limit: float | None) -> torch.Tensor:
    layer = SimpleNamespace(
        _exl3_inners=[
            {
                "gate": _FixedProjection(gate),
                "up": _FixedProjection(up),
                "down": _IdentityProjection(),
            }
        ]
    )
    return exl3.apply_exl3_python_loop(
        torch.zeros(1, 1),
        torch.tensor([[0]], dtype=torch.long),
        torch.tensor([[1.0]]),
        layer._exl3_inners,
        None,
        limit,
    )


def test_plain_and_clamped_swiglu_math() -> None:
    plain = _run_activation(20.0, 20.0, None)
    clamped = _run_activation(20.0, 20.0, 10.0)

    torch.testing.assert_close(plain, torch.tensor([[400.0]]), rtol=0, atol=1e-4)
    torch.testing.assert_close(
        clamped, torch.tensor([[99.9954605]]), rtol=0, atol=0.01
    )


@pytest.mark.parametrize(
    ("gate", "up", "limit"),
    [
        (9.99, 20.0, 10.0),
        (10.0, -20.0, 10.0),
        (10.01, -20.0, 10.0),
        (-10.01, 20.0, 10.0),
        (-9.99, -20.0, 10.0),
    ],
)
def test_clamped_swiglu_boundaries_and_negative_up(
    gate: float, up: float, limit: float
) -> None:
    output = _run_activation(gate, up, limit)
    expected = torch.nn.functional.silu(torch.tensor(min(gate, limit))) * min(
        max(up, -limit), limit
    )
    torch.testing.assert_close(output, expected.reshape(1, 1), rtol=0, atol=0.03)


def test_nonpositive_limit_keeps_plain_swiglu() -> None:
    expected = _run_activation(20.0, 20.0, None)
    for limit in (0.0, -1.0):
        torch.testing.assert_close(
            _run_activation(20.0, 20.0, limit), expected, rtol=0, atol=1e-4
        )


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"swiglu_limit": None}, None),
        ({"swiglu_limit": 0}, None),
        ({"swiglu_limit": -2}, None),
        ({"swiglu_limit": 7.5}, 7.5),
        ({"swiglu_limit": math.inf}, None),
        ({"swiglu_limit": math.nan}, None),
        ({}, None),
    ],
)
def test_moe_apply_resolves_swiglu_limit(
    monkeypatch: pytest.MonkeyPatch,
    attrs: dict[str, object],
    expected: float | None,
) -> None:
    calls: list[float | None] = []

    def record_apply(x, ids, weights, layer, *, limit, fused=None):
        del x, ids, weights, layer, fused
        calls.append(limit)
        return torch.zeros(1, 1)

    monkeypatch.setattr(exl3, "apply_exl3_experts", record_apply)
    method = object.__new__(exl3.Exl3MoEMethod)
    method.moe = SimpleNamespace(**attrs)

    output = method.apply(
        SimpleNamespace(),
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.long),
        None,
        None,
    )

    assert output.shape == (1, 1)
    assert calls == [expected]
