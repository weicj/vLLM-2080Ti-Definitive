"""Exl3LinearMethod: synthetic-pack checks that run without a model.

GPU test: random trellis / sign vectors for a fused three-shard column layer and
a single-matrix row layer, loaded through the real ``weight_loader`` closures
with vLLM's shard ids, then ``apply`` compared against exllamav3's own
``reconstruct_had_slice`` reference for rows 1 / 4 / 300 (kernel and
reconstruct paths).

CPU test: TP slicing helpers under tp_size=2 / tp_rank=1.

Run as a script on the serving box (no pytest needed):
    PYTHONPATH=src python tests/test_exl3_linear.py
"""

from __future__ import annotations

import sys

try:
    import pytest
    torch = pytest.importorskip("torch")
except ImportError:  # the serving venv has no pytest; run as a script instead

    class _Skip(Exception):
        pass

    class pytest:  # type: ignore[no-redef]
        @staticmethod
        def skip(msg):
            raise _Skip(msg)


MCG_MARKER = -877912083
MUL1_MARKER = -2082680531
K = 4
IN_FEATURES = 4096
SHARDS = [512, 512, 1024]


def _signs(n, gen, device):
    return torch.randn(n, generator=gen, device=device).sign().half()


def _rand_trellis(in_tiles, out_tiles, gen, device):
    return torch.randint(
        -(2**15), 2**15, (in_tiles, out_tiles, 16 * K), dtype=torch.int16,
        device=device, generator=gen,
    )


def _reference_weight(ext, trellis, suh, svh):
    """Original-basis W [in, out] exactly as LinearEXL3.reconstruct_hgemm builds it."""
    w = torch.empty((suh.numel(), svh.numel()), dtype=torch.half, device=trellis.device)
    ext.reconstruct_had_slice(w, trellis.contiguous(), suh, svh, K, True, False, 0)
    return w


def _patch_tp(monkeypatch_module, rank=0, size=1):
    monkeypatch_module.get_tensor_model_parallel_rank = lambda: rank
    monkeypatch_module.get_tensor_model_parallel_world_size = lambda: size


def _run_layer(method, in_features, shards, shard_ids, gen, device, ext):
    """Create weights, load each shard through its loader, build, and return (apply, ref)."""
    layer = torch.nn.Module()
    method.create_weights(
        layer, in_features, list(shards), in_features, sum(shards), torch.bfloat16,
        weight_loader=None,
    )
    for name in ("trellis", "suh", "svh", "mcg"):
        getattr(layer, name).data = getattr(layer, name).data.to(device)

    x = torch.randn(300, in_features, generator=gen, device=device).half()
    refs = []
    for sid, n_out in zip(shard_ids, shards):
        trellis = _rand_trellis(in_features // 16, n_out // 16, gen, device)
        suh = _signs(in_features, gen, device)
        svh = _signs(n_out, gen, device)
        mcg = torch.full((1,), MCG_MARKER, dtype=torch.int32, device=device)
        for name, t in (("trellis", trellis), ("suh", suh), ("svh", svh), ("mcg", mcg)):
            p = getattr(layer, name)
            if sid is None:
                p.weight_loader(p, t)  # plain ColumnParallel/RowParallel call shape
            else:
                p.weight_loader(p, t, sid)  # QKV / MergedColumn call shape (positional)
        refs.append(_reference_weight(ext, trellis, suh, svh))
    method.process_weights_after_loading(layer)
    assert getattr(layer, "trellis", None) is None, "staging params must be freed"
    w_ref = torch.cat(refs, dim=1)
    return layer, x, w_ref


def test_exl3_linear_basic():
    try:
        import exllamav3_ext as ext
    except ImportError:
        pytest.skip("exllamav3_ext not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import vllm.distributed as dist

    from vllm_exl3.exl3 import Exl3Config, Exl3LinearMethod

    _patch_tp(dist)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1234)
    cfg = Exl3Config(bits=2, non_routed_exl3={"modules": ["o_proj"], "bits": K})
    assert cfg._bits_for_non_routed("model.layers.3.self_attn.o_proj") == K
    assert cfg._matches_non_routed_exl3("model.layers.3.self_attn.o_proj")
    assert not cfg._matches_non_routed_exl3("model.layers.3.self_attn.xo_proj")
    assert cfg._bits_for_non_routed("model.layers.3.mlp.gate_up_proj") == 2  # unmatched -> base bits
    method = Exl3LinearMethod(cfg, bits=K)

    worst = 0.0
    for shards, ids in ((SHARDS, ["q", "k", "v"]), ([1024], [None])):
        layer, x, w_ref = _run_layer(method, IN_FEATURES, shards, ids, gen, device, ext)
        y_ref = (x.float() @ w_ref.float())
        for rows in (1, 4, 300):
            xin = x[:rows].to(torch.bfloat16)
            y = method.apply(layer, xin)
            assert y.dtype == torch.bfloat16 and y.shape == (rows, sum(shards)), y.shape
            err = (y.float() - y_ref[:rows]).norm() / y_ref[:rows].norm()
            worst = max(worst, err.item())
            print(f"EXL3_LINEAR shards={shards} rows={rows} rel_l2={err.item():.4e}")
            assert err < 2e-2, f"rows={rows} shards={shards}: rel err {err.item()}"
        # 3-D input and bias path
        xin3 = x[:6].view(2, 3, IN_FEATURES).to(torch.bfloat16)
        bias = torch.randn(sum(shards), device=device).to(torch.bfloat16)
        y3 = method.apply(layer, xin3, bias)
        assert y3.shape == (2, 3, sum(shards))
        err3 = ((y3.view(6, -1).float() - bias.float()) - y_ref[:6]).norm() / y_ref[:6].norm()
        assert err3 < 2e-2, err3.item()
    # sanity: a transposed / shuffled reference must NOT pass, or the check is vacuous
    y_bad = x[:4].float() @ w_ref.float()[torch.randperm(IN_FEATURES, generator=gen, device=device)]
    bad = (y_bad - y_ref[:4]).norm() / y_ref[:4].norm()
    assert bad > 0.5, f"reference check is too weak: {bad.item()}"
    print(f"EXL3_LINEAR_TEST PASS worst_rel_l2={worst:.4e}")


def test_exl3_linear_tp_slicing():
    """Column/row TP slicing through the real loaders on CPU (tp_size=2, tp_rank=1)."""
    try:
        import vllm.distributed as dist

        from vllm_exl3.exl3 import Exl3Config, Exl3LinearMethod
    except ImportError:
        pytest.skip("vllm / vllm_exl3 not importable")
    from vllm.model_executor.layers.linear import RowParallelLinear

    _patch_tp(dist, rank=1, size=2)
    try:
        cfg = Exl3Config(bits=K)
        method = Exl3LinearMethod(cfg, bits=K)
        gen = torch.Generator().manual_seed(7)
        full_in, full_out = 512, 256

        # column-parallel: this rank owns the upper half of N (trellis dim1, svh)
        layer = torch.nn.Module()
        method.create_weights(layer, full_in, [full_out // 2], full_in, full_out, torch.bfloat16)
        trellis = _rand_trellis(full_in // 16, full_out // 16, gen, "cpu")
        suh, svh = _signs(full_in, gen, "cpu"), _signs(full_out, gen, "cpu")
        layer.trellis.weight_loader(layer.trellis, trellis)
        layer.suh.weight_loader(layer.suh, suh)
        layer.svh.weight_loader(layer.svh, svh)
        assert torch.equal(layer.trellis.data, trellis[:, full_out // 32 :, :])
        assert torch.equal(layer.suh.data[0], suh)
        assert torch.equal(layer.svh.data, svh[full_out // 2 :])

        # row-parallel: this rank owns the upper half of K (trellis dim0, suh)
        layer = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(layer)
        method.create_weights(layer, full_in // 2, [full_out], full_in, full_out, torch.bfloat16)
        layer.trellis.weight_loader(layer.trellis, trellis)
        layer.suh.weight_loader(layer.suh, suh)
        layer.svh.weight_loader(layer.svh, svh)
        assert torch.equal(layer.trellis.data, trellis[full_in // 32 :, :, :])
        assert torch.equal(layer.suh.data[0], suh[full_in // 2 :])
        assert torch.equal(layer.svh.data, svh)
        print("EXL3_LINEAR_TP_TEST PASS")
    finally:
        _patch_tp(dist)


def test_exl3_linear_mixed_mul1():
    """6-shard mixed layer with bf16_shards [3,4,5] and mul1 codebook."""
    try:
        import exllamav3_ext as ext
    except ImportError:
        pytest.skip("exllamav3_ext not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import vllm.distributed as dist

    from vllm_exl3.exl3 import Exl3Config, Exl3LinearMethod

    _patch_tp(dist)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(4321)

    # Config with layers dict specifying bf16_shards and mul1 codebook
    cfg = Exl3Config(
        bits=4,
        non_routed_exl3={
            "codebook": "mul1",
            "layers": {
                "model.language_model.layers.0.mlp.gate_up_proj": {
                    "bits": K,
                    "bf16_shards": [3, 4, 5]
                }
            }
        }
    )

    # Verify config parsing
    prefix = "model.language_model.layers.0.mlp.gate_up_proj"
    assert cfg._matches_non_routed_exl3(prefix), f"prefix {prefix} should match"
    assert cfg._bits_for_non_routed(prefix) == K
    assert cfg._bf16_shards_for(prefix) == [3, 4, 5]
    # Unlisted prefix falls back to base bits and no bf16_shards
    assert cfg._bits_for_non_routed("model.layers.0.mlp.down_proj") == cfg.bits
    assert cfg._bf16_shards_for("model.layers.0.mlp.down_proj") == []

    method = Exl3LinearMethod(cfg, bits=K)
    shards = [512, 512, 512, 64, 128, 128]  # 6 shards, last 3 are bf16
    bf16_shards = [3, 4, 5]

    layer = torch.nn.Module()
    layer.prefix = prefix
    method.create_weights(
        layer, IN_FEATURES, shards, IN_FEATURES, sum(shards), torch.bfloat16,
        weight_loader=None,
    )
    for name in ("trellis", "suh", "svh", "mcg", "mul1", "weight"):
        if hasattr(layer, name):
            getattr(layer, name).data = getattr(layer, name).data.to(device)

    x = torch.randn(300, IN_FEATURES, generator=gen, device=device).half()

    # Generate all tensors upfront
    shard_tensors = []
    for sid, n_out in enumerate(shards):
        trellis = _rand_trellis(IN_FEATURES // 16, n_out // 16, gen, device)
        suh = _signs(IN_FEATURES, gen, device)
        svh = _signs(n_out, gen, device)
        if sid in bf16_shards:
            w_bf16 = torch.randn(n_out, IN_FEATURES, dtype=torch.bfloat16, device=device)
        else:
            w_bf16 = None
        shard_tensors.append((trellis, suh, svh, w_bf16))

    # Load each shard: mul1 marker for EXL3, stale BF16 weight, then bf16 shard weights
    for sid, n_out in enumerate(shards):
        trellis, suh, svh, w_bf16 = shard_tensors[sid]

        # For EXL3 shards (0-2): use mul1 marker; for bf16 (3-5): use 0
        if sid in bf16_shards:
            mul1 = torch.full((1,), 0, dtype=torch.int32, device=device)
            mcg = torch.full((1,), 0, dtype=torch.int32, device=device)
            # Stale BF16 weight (should be discarded)
            w_stale = torch.randn(n_out, IN_FEATURES, dtype=torch.bfloat16, device=device)
        else:
            mul1 = torch.full((1,), MUL1_MARKER, dtype=torch.int32, device=device)
            mcg = torch.full((1,), 0, dtype=torch.int32, device=device)
            w_stale = torch.randn(n_out, IN_FEATURES, dtype=torch.bfloat16, device=device)

        # Load through the real loaders (positional shard_id)
        layer.trellis.weight_loader(layer.trellis, trellis, sid)
        layer.suh.weight_loader(layer.suh, suh, sid)
        layer.svh.weight_loader(layer.svh, svh, sid)
        layer.mul1.weight_loader(layer.mul1, mul1, sid)
        layer.mcg.weight_loader(layer.mcg, mcg, sid)
        # Load stale BF16 weight (discarded for EXL3, kept for bf16)
        layer.weight.weight_loader(layer.weight, w_stale, sid)
        # For bf16 shards, load the real weight
        if w_bf16 is not None:
            layer.weight.weight_loader(layer.weight, w_bf16, sid)

    # Build LinearEXL3 for each EXL3 shard, keep bf16 weights
    method.process_weights_after_loading(layer)
    assert getattr(layer, "trellis", None) is None, "staging params must be freed"
    assert getattr(layer, "mul1", None) is None
    assert getattr(layer, "mcg", None) is None
    assert hasattr(layer, "_exl3_bf16_weight"), "bf16 weight should be kept"

    # Build reference output using reconstruct_had_slice for EXL3 + dense for bf16
    refs = []
    for sid, n_out in enumerate(shards):
        trellis, suh, svh, w_bf16 = shard_tensors[sid]
        if sid in bf16_shards:
            # bf16 weight is (out, in), transpose to (in, out) for reference
            refs.append(w_bf16.t())
        else:
            # EXL3: reconstruct with mul1 codebook
            w = torch.empty((suh.numel(), svh.numel()), dtype=torch.half, device=device)
            ext.reconstruct_had_slice(w, trellis.contiguous(), suh, svh, K, False, True, 0)
            refs.append(w)
    w_ref = torch.cat(refs, dim=1)

    # Apply and compare
    y_ref = (x.float() @ w_ref.float())
    for rows in (1, 4, 300):
        xin = x[:rows].to(torch.bfloat16)
        y = method.apply(layer, xin)
        assert y.dtype == torch.bfloat16 and y.shape == (rows, sum(shards)), y.shape
        err = (y.float() - y_ref[:rows]).norm() / y_ref[:rows].norm()
        print(f"EXL3_LINEAR_MIXED rows={rows} rel_l2={err.item():.4e}")
        assert err < 2e-2, f"rows={rows}: rel err {err.item()}"
    print("EXL3_LINEAR_MIXED_TEST PASS")


if __name__ == "__main__":
    failed = False
    for fn in (test_exl3_linear_tp_slicing, test_exl3_linear_basic, test_exl3_linear_mixed_mul1):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "_Skip":
                print(f"SKIP {fn.__name__}: {e}")
                continue
            import traceback

            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {e}")
            failed = True
    sys.exit(1 if failed else 0)
