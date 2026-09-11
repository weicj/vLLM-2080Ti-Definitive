"""CPU regression tests for EXL3 expert-map and TP loader geometry."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class _ReadOnlyExpertMapLayer:
    def __init__(self, expert_map: torch.Tensor) -> None:
        self._raw_expert_map = expert_map

    @property
    def expert_map(self) -> torch.Tensor:
        return self._raw_expert_map


class _MoEOwner:
    tp_rank = 0
    tp_size = 1

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id


def test_moe_loader_prefers_layer_tp_geometry() -> None:
    """A MoE layer with TP=1 must not inherit process TP=8 slicing."""
    owner = _MoEOwner()
    param = torch.nn.Parameter(
        torch.empty(1, 2, 1, 1, 32, dtype=torch.int16), requires_grad=False
    )
    param._exl3_owner = owner
    loaded = torch.arange(32, dtype=torch.int16).reshape(1, 1, 32)
    method = object.__new__(exl3.Exl3MoEMethod)
    method._load_exl3(
        param,
        loaded,
        "experts.w13_trellis",
        shard_id="w1",
        expert_id=0,
    )

    torch.testing.assert_close(param[0, 0], loaded)


def test_moe_loader_uses_ep_local_geometry_before_process_tp() -> None:
    """EP experts remain whole even when the enclosing model uses TP=4."""
    owner = SimpleNamespace(
        moe_config=SimpleNamespace(tp_rank=0, tp_size=1),
    )

    assert exl3._resolve_tp_geometry(owner) == (0, 1)


def test_non_128_moe_fails_before_a_kernel_can_corrupt_results() -> None:
    """TP=4's 160-wide experts need EP rather than an invalid EXL3 kernel."""
    with pytest.raises(RuntimeError, match="128-aligned"):
        exl3.apply_exl3_reconstruct_moe(
            torch.empty(1, 16),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1),
            [],
            None,
        )


def test_pin_expert_map_uses_private_cache_for_read_only_property() -> None:
    raw = torch.tensor([2, 0, 1], dtype=torch.int32)
    layer = _ReadOnlyExpertMapLayer(raw)

    pinned = exl3.pin_exl3_expert_map(layer, torch.device("cpu"))

    assert pinned is not None
    assert pinned.dtype == torch.long
    assert pinned.device == torch.device("cpu")
    assert layer._exl3_pinned_expert_map is pinned
    assert exl3.pin_exl3_expert_map(layer, torch.device("cpu")) is pinned

    layer._raw_expert_map = torch.tensor([1, 2, 0], dtype=torch.int64)
    refreshed = exl3.pin_exl3_expert_map(layer, torch.device("cpu"))
    assert refreshed is not pinned
    torch.testing.assert_close(refreshed, layer._raw_expert_map)


@pytest.mark.parametrize(
    ("is_row_parallel", "tp_rank", "expected"),
    [
        (True, 1, torch.arange(256, dtype=torch.float32).reshape(16, 16)[:, 8:]),
        (False, 1, torch.arange(256, dtype=torch.float32).reshape(16, 16)[8:, :]),
    ],
)
def test_dense_bf16_loader_slices_correct_tp_axis(
    is_row_parallel: bool,
    tp_rank: int,
    expected: torch.Tensor,
) -> None:
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=tp_rank, tp_size=2)
    param_shape = (16, 8) if is_row_parallel else (8, 16)
    param = torch.nn.Parameter(torch.zeros(param_shape), requires_grad=False)
    output_sizes = [16] if is_row_parallel else [8]
    loader = method._make_weight_loader(
        "weight", 1, output_sizes, is_row_parallel, [0], layer, False
    )
    loaded = torch.arange(256, dtype=torch.float32).reshape(16, 16)

    loader(param, loaded)

    torch.testing.assert_close(param, expected)


def test_qkv_loader_replicated_kv_heads_uses_shard_specific_tp() -> None:
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=7, tp_size=8)
    output_sizes = [512, 128, 128]
    total_tiles = sum(output_sizes) // 16
    param = torch.nn.Parameter(
        torch.zeros(1, total_tiles, 32, dtype=torch.int16), requires_grad=False
    )
    loader = method._make_weight_loader(
        "trellis", 3, output_sizes, False, [], layer, True
    )
    loaded = torch.arange(1 * (128 // 16) * 32, dtype=torch.int16).reshape(
        1, 128 // 16, 32
    )

    loader(param, loaded, loaded_shard_id="k")

    torch.testing.assert_close(param[:, 512 // 16 : (512 + 128) // 16], loaded)

    svh = torch.nn.Parameter(
        torch.zeros(sum(output_sizes), dtype=torch.float16), requires_grad=False
    )
    svh_loader = method._make_weight_loader(
        "svh", 3, output_sizes, False, [], layer, True
    )
    loaded_svh = torch.arange(128, dtype=torch.float16)
    svh_loader(svh, loaded_svh, loaded_shard_id="k")
    torch.testing.assert_close(svh[512:640], loaded_svh)


def test_padded_merged_column_shards_zero_extend_independently() -> None:
    """Flash shared-expert gate/up shards are 320-wide, not 128-aligned."""
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=0, tp_size=1, _exl3_linear_padded=True)

    # Two checkpoint shards of 320 columns are loaded into independently
    # padded 384-column destinations. The tail must stay zero, rather than
    # leaking uninitialised trellis or scale entries into the trimmed output.
    svh = torch.nn.Parameter(torch.zeros(768, dtype=torch.float16), requires_grad=False)
    svh_loader = method._make_weight_loader(
        "svh", 2, [384, 384], False, [], layer, False
    )
    first = torch.arange(320, dtype=torch.float16)
    second = torch.arange(320, 640, dtype=torch.float16)
    svh_loader(svh, first, loaded_shard_id=0)
    svh_loader(svh, second, loaded_shard_id=1)

    torch.testing.assert_close(svh[:320], first)
    torch.testing.assert_close(svh[384:704], second)
    assert not torch.count_nonzero(svh[320:384])
    assert not torch.count_nonzero(svh[704:768])


def test_padded_merged_column_shards_support_tensor_parallel_loading() -> None:
    """Padding happens after rank-local column TP slicing, for each shard."""
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(tp_rank=1, tp_size=2, _exl3_linear_padded=True)
    svh = torch.nn.Parameter(torch.zeros(768, dtype=torch.float16), requires_grad=False)
    loader = method._make_weight_loader("svh", 2, [384, 384], False, [], layer, False)

    gate = torch.arange(640, dtype=torch.float16)
    up = torch.arange(1000, 1640, dtype=torch.float16)
    loader(svh, gate, loaded_shard_id=0)
    loader(svh, up, loaded_shard_id=1)

    torch.testing.assert_close(svh[:320], gate[320:])
    torch.testing.assert_close(svh[384:704], up[320:])
    assert not torch.count_nonzero(svh[320:384])
    assert not torch.count_nonzero(svh[704:768])


def test_padded_replicated_kv_shard_is_not_sliced_twice_under_tp() -> None:
    """GQA K/V tensors may already be local while the process uses TP=2."""
    method = object.__new__(exl3.Exl3LinearMethod)
    layer = SimpleNamespace(
        tp_rank=1,
        tp_size=2,
        _exl3_linear_padded=True,
        _exl3_linear_true_in=2560,
        _exl3_linear_true_out=[320],
    )
    svh = torch.nn.Parameter(torch.zeros(384, dtype=torch.float16), requires_grad=False)
    loader = method._make_weight_loader("svh", 1, [384], False, [], layer, False)
    local_kv = torch.arange(320, dtype=torch.float16)

    loader(svh, local_kv)

    torch.testing.assert_close(svh[:320], local_kv)
    assert not torch.count_nonzero(svh[320:])


def test_qkv_global_fused_tensor_is_split_before_tensor_parallel_loading() -> None:
    """QKV mapper passes a global fused tensor and shard ids as one tuple."""
    method = object.__new__(exl3.Exl3LinearMethod)
    local_sizes = [2560, 1280, 1280]
    layer = SimpleNamespace(
        tp_rank=1,
        tp_size=2,
        _exl3_linear_padded=False,
        _exl3_linear_true_out=local_sizes,
    )
    param = torch.nn.Parameter(torch.zeros(sum(local_sizes)), requires_grad=False)
    loader = method._make_weight_loader("svh", 3, local_sizes, False, [], layer, True)
    loaded = torch.arange(10240, dtype=torch.float32)

    loader(param, loaded, loaded_shard_id=(0, 1, 2))

    expected = torch.cat((loaded[2560:5120], loaded[6400:7680], loaded[8960:10240]))
    torch.testing.assert_close(param, expected)
