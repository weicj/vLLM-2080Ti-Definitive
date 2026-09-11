"""CPU tests for the dense row-count routing that keeps 17..144-row calls off the cooperative GEMM."""

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    def forward(self, x, params, out_dtype):
        self.calls.append((int(x.shape[0]), dict(params)))
        return x.float()[:, :4]


def _run(rows: int):
    lin = _Recorder()
    x = torch.arange(rows * 8, dtype=torch.float16).reshape(rows, 8)
    y = exl3._dense_forward(lin, x)
    return lin.calls, y


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16])
def test_small_row_counts_keep_the_native_dispatch(rows, monkeypatch) -> None:
    monkeypatch.setattr(exl3, "_EXL3_COOP_GEMM", False)
    monkeypatch.setattr(exl3, "_EXL3_RECON_MIN_ROWS", 17)
    calls, y = _run(rows)
    assert calls == [(rows, {})]
    assert y.shape == (rows, 4)


@pytest.mark.parametrize("rows", [17, 72, 128, 144])
def test_mid_range_rows_reconstruct(rows, monkeypatch) -> None:
    monkeypatch.setattr(exl3, "_EXL3_COOP_GEMM", False)
    monkeypatch.setattr(exl3, "_EXL3_RECON_MIN_ROWS", 17)
    calls, y = _run(rows)
    assert calls == [(rows, {"reconstruct": True})]
    assert y.shape == (rows, 4)


def test_above_threshold_leaves_the_decision_to_exllamav3(monkeypatch) -> None:
    monkeypatch.setattr(exl3, "_EXL3_COOP_GEMM", False)
    monkeypatch.setattr(exl3, "_EXL3_RECON_MIN_ROWS", 17)
    calls, _ = _run(145)
    assert calls == [(145, {})]


def test_coop_gemm_override_restores_old_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(exl3, "_EXL3_COOP_GEMM", True)
    monkeypatch.setattr(exl3, "_EXL3_RECON_MIN_ROWS", 17)
    calls, _ = _run(72)
    assert calls == [(72, {})]


def test_threshold_override(monkeypatch) -> None:
    monkeypatch.setattr(exl3, "_EXL3_COOP_GEMM", False)
    monkeypatch.setattr(exl3, "_EXL3_RECON_MIN_ROWS", 9)
    assert _run(9)[0] == [(9, {"reconstruct": True})]
    assert _run(8)[0] == [(8, {})]


def test_env_int_parsing(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", raising=False)
    assert exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 17) == 17
    monkeypatch.setenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", "33")
    assert exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 17) == 33
    monkeypatch.setenv("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", "abc")
    assert exl3._env_int("VLLM_EXL3_RECONSTRUCT_MIN_ROWS", 17) == 17
