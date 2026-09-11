"""CPU regression tests for reachable ExLlamaV3 fat-expert execution."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


def _pack(*, k_words: int = 64, suh_value: float = 1.0):
    return SimpleNamespace(
        in_features=16,
        out_features=16,
        K=k_words // 16,
        mcg=True,
        mul1=False,
        trellis=torch.zeros(1, 1, k_words, dtype=torch.int16),
        suh=torch.full((16,), suh_value, dtype=torch.float16),
        svh=torch.ones(16, dtype=torch.float16),
    )


def test_fat_dispatch_is_reachable_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[int], int]] = []
    fused_args: list[tuple[object, ...]] = []

    class _Exllama:
        def exl3_moe(self, *args):
            fused_args.append(args)
            return None

    def record_fat(
        xh,
        token_sorted,
        weight_sorted,
        counts,
        inners,
        limit,
        cap,
        out,
    ):
        del xh, weight_sorted, inners, limit
        calls.append((counts, cap))
        out.add_(1.0)
        assert token_sorted.tolist() == [0, 1]
        return out

    monkeypatch.setattr(exl3, "FAT_EXPERT_THRESHOLD", 1)
    monkeypatch.setattr(exl3, "_exl3_moe_accepts_num_active", lambda _: True)
    monkeypatch.setattr(exl3, "apply_exl3_batched_fat", record_fat)
    monkeypatch.setitem(sys.modules, "exllamav3_ext", _Exllama())

    layer = SimpleNamespace(
        _exl3_ptrs={
            key: object()
            for key in (
                "gate_trellis",
                "gate_suh",
                "gate_svh",
                "up_trellis",
                "up_suh",
                "up_svh",
                "down_trellis",
                "down_suh",
                "down_svh",
            )
        },
        _exl3_fused_temps=(None,) * 5,
    )
    output = exl3.apply_exl3_fused_moe(
        torch.zeros(2, 16),
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1),
        layer,
        [{}],
        None,
    )

    assert calls == [([2], 1)]
    assert len(fused_args) == 1
    assert len(fused_args[0]) == 32
    assert fused_args[0][-2:] == (-1, False)
    torch.testing.assert_close(output, torch.ones(2, 16))


def test_fat_path_uses_exllamav3_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exllama_calls: list[str] = []

    class _Exllama:
        def had_r_128(self, source, destination, *args):
            del args
            exllama_calls.append("had_r_128")
            destination.copy_(source)

        def reconstruct(self, destination, *args):
            del args
            exllama_calls.append("reconstruct")
            destination.zero_()

        def hgemm(self, _a, _b, destination):
            exllama_calls.append("hgemm")
            destination.zero_()

    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: _Exllama())

    pack = _pack(k_words=32)
    out = exl3.apply_exl3_batched_fat(
        torch.zeros(257, 16, dtype=torch.float16),
        torch.arange(257, dtype=torch.long),
        torch.ones(257, dtype=torch.float16),
        [257],
        [{"gate": pack, "up": pack, "down": pack}],
        None,
        256,
        torch.zeros(257, 16),
    )

    assert out.shape == (257, 16)
    assert "had_r_128" in exllama_calls
    assert "reconstruct" in exllama_calls
    assert "hgemm" in exllama_calls


def test_fat_scratch_keys_include_k_words_and_bucket_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exl3, "_FAT_SCRATCH_CACHE", {})
    gate_k4 = _pack(k_words=64)
    gate_k2 = _pack(k_words=32)

    first = exl3._fat_scratch(torch.device("cpu"), 257, gate_k4)
    same_bucket = exl3._fat_scratch(torch.device("cpu"), 260, gate_k4)
    other_bitrate = exl3._fat_scratch(torch.device("cpu"), 257, gate_k2)

    assert first is same_bucket
    assert first is not other_bitrate
    assert ("cpu", 512, 16, 16, 64) in exl3._FAT_SCRATCH_CACHE
    assert ("cpu", 512, 16, 16, 32) in exl3._FAT_SCRATCH_CACHE


def test_distinct_gate_and_up_suh_transforms_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transforms: list[torch.Tensor | None] = []

    class _Exllama:
        def had_r_128(self, source, destination, suh, *args):
            del args
            transforms.append(suh)
            destination.copy_(source)

        def reconstruct(self, destination, *args):
            del args
            destination.zero_()

        def hgemm(self, _a, _b, destination):
            destination.zero_()

    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: _Exllama())
    gate = _pack(k_words=32, suh_value=1.0)
    up = _pack(k_words=32, suh_value=2.0)
    down = _pack(k_words=32, suh_value=3.0)

    exl3.apply_exl3_batched_fat(
        torch.zeros(257, 16, dtype=torch.float16),
        torch.arange(257, dtype=torch.long),
        torch.ones(257, dtype=torch.float16),
        [257],
        [{"gate": gate, "up": up, "down": down}],
        None,
        256,
        torch.zeros(257, 16),
    )

    assert torch.equal(transforms[0], gate.suh)
    assert torch.equal(transforms[1], up.suh)
