# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small FP16 GEMV used by Qwen4-Exp single-token decode on SM75."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the runtime environment
    triton = None
    tl = None

# vLLM can disable Triton after its backend/driver probe even when the Python
# package is importable.  Treat that state as unavailable for this kernel too.
if not HAS_TRITON:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gemv_m1_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        k_size: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k_base in tl.range(0, k_size, BLOCK_K):
            k = k_base + offsets
            mask = k < k_size
            x = tl.load(x_ptr + k, mask=mask, other=0.0)
            weight = tl.load(
                weight_ptr + row * k_size + k,
                mask=mask,
                other=0.0,
            )
            acc += x.to(tl.float32) * weight.to(tl.float32)
        tl.store(out_ptr + row, tl.sum(acc, axis=0).to(tl.float16))

    @triton.jit
    def _gemv_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        n_rows,
        n_cols,
        stride_x_row,
        stride_x_col,
        stride_w_row,
        stride_w_col,
        stride_out_row,
        stride_out_col,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k_base in tl.range(0, n_cols, BLOCK_K):
            k = k_base + offsets
            mask = k < n_cols
            x = tl.load(
                x_ptr + row * stride_x_row + k * stride_x_col,
                mask=mask,
                other=0.0,
            )
            weight = tl.load(
                weight_ptr + col * stride_w_row + k * stride_w_col,
                mask=mask,
                other=0.0,
            )
            acc += x.to(tl.float32) * weight.to(tl.float32)
        tl.store(
            out_ptr + row * stride_out_row + col * stride_out_col,
            tl.sum(acc, axis=0).to(tl.float16),
        )


def is_available() -> bool:
    return HAS_TRITON and triton is not None


def gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T`` for contiguous FP16 2-D tensors."""
    if not is_available():
        raise RuntimeError("Triton is not available")
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("invalid GEMV shapes")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("SM75 Triton GEMV currently requires FP16")
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        raise ValueError("GEMV tensors must be on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("GEMV tensors must be contiguous")

    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    block_k = 512 if weight.shape[1] >= 8192 else 256
    if x.shape[0] == 1:
        _gemv_m1_kernel[(weight.shape[0],)](
            x,
            weight,
            output,
            n_cols=weight.shape[0],
            k_size=weight.shape[1],
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
    else:
        _gemv_kernel[(x.shape[0], weight.shape[0])](
            x,
            weight,
            output,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
    return output
