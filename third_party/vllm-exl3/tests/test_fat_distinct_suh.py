"""CPU regression test for the fat-expert prefill path with distinct gate/up input rotations.

Packs that quantize gate_proj and up_proj separately carry different ``suh`` vectors, so
``apply_exl3_batched_fat`` takes its distinct-suh branch. That branch used to hand column slices
of the shared ``gate_up`` scratch buffer to ``ext.hgemm`` and ``ext.had_r_128``; both extension
kernels index contiguous row-major operands, so the expert output was uncorrelated with the
reference. The fake extension below refuses non-contiguous operands, which is exactly what the
real kernels silently mis-read, and the pure-torch reference uses the same fake arithmetic.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3

HIDDEN = 256
INTER = 128
K = 3
N_ROWS = 300  # above FAT cap of 256, so the fat path runs
CAP = 256


class _FakeExt:
    """Deterministic stand-in for exllamav3_ext that rejects strided operands."""

    def __init__(self) -> None:
        self.hgemm_out_shapes: list[tuple[int, ...]] = []

    @staticmethod
    def _contiguous(name: str, *tensors) -> None:
        for t in tensors:
            if t is not None and not t.is_contiguous():
                raise AssertionError(f"{name}: non-contiguous operand {tuple(t.shape)} {t.stride()}")

    @staticmethod
    def weight_for(trellis: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(trellis.to(torch.int64).sum().item()) & 0xFFFF)
        return (torch.randn(shape, generator=gen) * 0.05).to(torch.float16)

    def reconstruct(self, w, trellis, k, mcg, mul1) -> None:
        self._contiguous("reconstruct", w, trellis)
        w.copy_(self.weight_for(trellis, tuple(w.shape)))

    def hgemm(self, a, b, c) -> None:
        self._contiguous("hgemm", a, b, c)
        self.hgemm_out_shapes.append(tuple(c.shape))
        c.copy_((a.float() @ b.float()).to(c.dtype))

    def had_r_128(self, x, out, suh, svh, scale) -> None:
        self._contiguous("had_r_128", x, out, suh, svh)
        y = x.float()
        if suh is not None:
            y = y * suh.float()
        if svh is not None:
            y = y * svh.float()
        out.copy_((y * float(scale)).to(out.dtype))


def _projection(in_features: int, out_features: int, seed: int) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        trellis=torch.randint(-3, 4, (in_features // 16, out_features // 16, 16 * K), generator=gen, dtype=torch.int16),
        suh=(torch.rand(in_features, generator=gen) + 0.5).to(torch.float16),
        svh=(torch.rand(out_features, generator=gen) + 0.5).to(torch.float16),
        K=K,
        mcg=False,
        mul1=True,
        in_features=in_features,
        out_features=out_features,
    )


def _reference(ext: _FakeExt, xh, gate, up, down, weight: float) -> torch.Tensor:
    w_gate = ext.weight_for(gate.trellis, (HIDDEN, INTER))
    w_up = ext.weight_for(up.trellis, (HIDDEN, INTER))
    w_down = ext.weight_for(down.trellis, (INTER, HIDDEN))
    gate_h = (xh.float() * gate.suh.float()).to(torch.float16)
    up_h = (xh.float() * up.suh.float()).to(torch.float16)
    g = (gate_h.float() @ w_gate.float()) * gate.svh.float()
    u = (up_h.float() @ w_up.float()) * up.svh.float()
    act = torch.sigmoid(g) * g * u
    act_h = act.to(torch.float16)
    h2 = (act_h.float() * down.suh.float()).to(torch.float16)
    d = (h2.float() @ w_down.float()) * down.svh.float()
    return d * weight


def test_distinct_suh_fat_branch_matches_reference(monkeypatch) -> None:
    ext = _FakeExt()
    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: ext)
    exl3._FAT_SCRATCH_CACHE.clear()

    gate = _projection(HIDDEN, INTER, seed=1)
    up = _projection(HIDDEN, INTER, seed=2)
    down = _projection(INTER, HIDDEN, seed=3)
    assert not torch.equal(gate.suh, up.suh)

    gen = torch.Generator().manual_seed(7)
    xh = (torch.randn(N_ROWS, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    token_sorted = torch.arange(N_ROWS, dtype=torch.int64)
    weight_sorted = torch.full((N_ROWS,), 0.5, dtype=torch.float32)
    out = torch.zeros(N_ROWS, HIDDEN, dtype=torch.float32)

    exl3.apply_exl3_batched_fat(
        xh,
        token_sorted,
        weight_sorted,
        [N_ROWS],
        [{"gate": gate, "up": up, "down": down}],
        None,
        CAP,
        out,
    )

    ref = _reference(ext, xh, gate, up, down, 0.5)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    # gate and up GEMMs must land in contiguous [rows, intermediate] temporaries
    assert (N_ROWS, INTER) in ext.hgemm_out_shapes
    exl3._FAT_SCRATCH_CACHE.clear()


def test_shared_suh_fat_branch_still_matches_reference(monkeypatch) -> None:
    ext = _FakeExt()
    monkeypatch.setattr(exl3, "load_exllamav3_ext", lambda: ext)
    exl3._FAT_SCRATCH_CACHE.clear()

    gate = _projection(HIDDEN, INTER, seed=11)
    up = _projection(HIDDEN, INTER, seed=12)
    up.suh = gate.suh.clone()
    down = _projection(INTER, HIDDEN, seed=13)

    gen = torch.Generator().manual_seed(8)
    xh = (torch.randn(N_ROWS, HIDDEN, generator=gen) * 0.5).to(torch.float16)
    token_sorted = torch.arange(N_ROWS, dtype=torch.int64)
    weight_sorted = torch.full((N_ROWS,), 0.5, dtype=torch.float32)
    out = torch.zeros(N_ROWS, HIDDEN, dtype=torch.float32)

    exl3.apply_exl3_batched_fat(
        xh, token_sorted, weight_sorted, [N_ROWS], [{"gate": gate, "up": up, "down": down}], None, CAP, out
    )

    # the shared branch reconstructs one fused [hidden, 2*inter] weight from the concatenated trellis,
    # so build the reference from that same fused weight
    packed13 = torch.cat([gate.trellis, up.trellis], dim=1).contiguous()
    w13 = ext.weight_for(packed13, (HIDDEN, 2 * INTER))
    w_down = ext.weight_for(down.trellis, (INTER, HIDDEN))
    h13 = (xh.float() * gate.suh.float()).to(torch.float16)
    svh13 = torch.cat([gate.svh, up.svh]).float()
    gu = (h13.float() @ w13.float()) * svh13
    g, u = gu[:, :INTER], gu[:, INTER:]
    act = torch.sigmoid(g) * g * u
    h2 = (act.to(torch.float16).float() * down.suh.float()).to(torch.float16)
    ref = (h2.float() @ w_down.float()) * down.svh.float() * 0.5
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    exl3._FAT_SCRATCH_CACHE.clear()
