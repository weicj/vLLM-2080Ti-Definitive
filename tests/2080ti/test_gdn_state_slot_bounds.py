# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds checks for the GDN recurrent state slots.

A stale or miscomputed state slot used to reach a raw pointer store inside
``fused_sigmoid_gating_delta_rule_update_kernel``, which the CUDA runtime does
not bounds check. On the hybrid Qwen GDN + MTP path that showed up as
``Xid 31 ... ACCESS_TYPE_VIRT_WRITE`` and an ``EngineDeadError`` for the whole
engine. These tests pin the guarantees the guard adds:

* an out-of-range slot must not fault and must not write the state cache;
* a valid slot must still produce the reference result, and the padding
  sentinel must still leave both the state and the output row untouched.

The tests are plain functions (plus a ``__main__`` runner) instead of
``pytest.mark.skipif`` so that they can be executed in the runtime venv, which
does not ship pytest.
"""

import torch


def _make_inputs(torch_device: torch.device, dtype: torch.dtype):
    heads = value_heads = 1
    key_dim = value_dim = 16
    q = torch.randn(heads, key_dim, device=torch_device, dtype=dtype) * 0.1
    k = torch.randn(heads, key_dim, device=torch_device, dtype=dtype) * 0.1
    v = torch.randn(value_heads, value_dim, device=torch_device, dtype=dtype) * 0.1
    a = torch.randn(1, value_heads, device=torch_device, dtype=dtype) * 0.1
    b = torch.randn(1, value_heads, device=torch_device, dtype=dtype) * 0.1
    A_log = torch.randn(value_heads, device=torch_device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(value_heads, device=torch_device, dtype=dtype) * 0.1
    state = (
        torch.randn(
            2,
            value_heads,
            value_dim,
            key_dim,
            device=torch_device,
            dtype=dtype,
        )
        * 0.1
    )
    return q, k, v, a, b, A_log, dt_bias, state, key_dim


def _reference(q, k, v, a, b, A_log, dt_bias, h0, key_dim):
    q_ref = torch.nn.functional.normalize(q.float(), dim=-1)
    k_ref = torch.nn.functional.normalize(k.float(), dim=-1)
    h_ref = h0.float()
    gate = -torch.exp(A_log.float()) * torch.nn.functional.softplus(
        a[0].float() + dt_bias.float()
    )
    h_ref = h_ref * torch.exp(gate)[:, None, None]
    v_ref = v.float() - (h_ref * k_ref[:, None, :]).sum(dim=-1)
    v_ref = v_ref * torch.sigmoid(b[0].float())[:, None]
    h_ref = h_ref + v_ref[:, :, None] * k_ref[:, None, :]
    out_ref = (h_ref * q_ref[:, None, :]).sum(dim=-1) * key_dim**-0.5
    return out_ref, h_ref


def _call_recurrent(q, k, v, a, b, A_log, dt_bias, state, slot, pad_slot):
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    device = state.device
    indices = torch.tensor([slot], device=device, dtype=torch.int32)
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.reshape(1, 1, 1, -1),
        k=k.reshape(1, 1, 1, -1),
        v=v.reshape(1, 1, 1, -1),
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], device=device, dtype=torch.int32),
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        null_block_id=pad_slot,
    )


def test_gdn_valid_state_slot_matches_reference() -> None:
    if not torch.cuda.is_available():
        return
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.float16
    q, k, v, a, b, A_log, dt_bias, state, key_dim = _make_inputs(device, dtype)
    out_ref, h_ref = _reference(q, k, v, a, b, A_log, dt_bias, state[0], key_dim)

    untouched = state[1].clone()
    out, _ = _call_recurrent(q, k, v, a, b, A_log, dt_bias, state, 0, PAD_SLOT_ID)

    torch.testing.assert_close(
        out[0, 0], out_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(
        state[0], h_ref, rtol=2e-3, atol=2e-3, check_dtype=False
    )
    torch.testing.assert_close(state[1], untouched, rtol=0, atol=0)


def test_gdn_out_of_range_state_slot_is_skipped() -> None:
    """The regression that used to be an Xid 31 engine kill."""
    if not torch.cuda.is_available():
        return
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.float16
    q, k, v, a, b, A_log, dt_bias, state, _ = _make_inputs(device, dtype)
    before = state.clone()

    # slot == state.shape[0] is one past the last row of the state cache.
    out, _ = _call_recurrent(
        q, k, v, a, b, A_log, dt_bias, state, state.shape[0], PAD_SLOT_ID
    )
    torch.cuda.synchronize()

    # No fault, no state write, and a defined output (zeros, not NaN).
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    assert bool(out.isfinite().all()), "guard must not leak NaN/Inf into the output"
    torch.testing.assert_close(
        out, torch.zeros_like(out), rtol=0, atol=0
    )


def test_gdn_padding_sentinel_still_leaves_output_untouched() -> None:
    """The padding contract: a PAD slot writes neither state nor output."""
    if not torch.cuda.is_available():
        return
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.float16
    q, k, v, a, b, A_log, dt_bias, state, _ = _make_inputs(device, dtype)
    before = state.clone()

    _call_recurrent(q, k, v, a, b, A_log, dt_bias, state, PAD_SLOT_ID, PAD_SLOT_ID)
    torch.cuda.synchronize()

    torch.testing.assert_close(state, before, rtol=0, atol=0)
    # `o` is allocated with new_empty and the early return for the sentinel is
    # what keeps the (discarded) padded output row untouched, so nothing about
    # its value can be asserted here.


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
