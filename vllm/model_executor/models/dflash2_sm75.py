# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 numerical codec for BF16-trained Qwen DFlash2 drafts.

The DFlash2 checkpoint has BF16 activation boundaries, while a Turing worker
executes its dense layers in FP16.  The codec makes that cross-dtype contract
explicit: activation values are rounded at the trained boundaries, the MLP
wire payload is row-scaled before FP16 storage, and the residual state remains
in FP32.  It is deliberately a DFlash2-only SM75 path.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton

_ACTIVATION_GAIN = 32.0
_RESIDUAL_GAIN = 256.0
_HALF_PAYLOAD_LIMIT = 32752.0
# Weight dtypes that are applied as-is by the linear kernel. Note that
# ``float8`` dtypes report ``is_floating_point == True`` even though they are
# 8-bit payloads that need their scale applied, so list the dense ones out.
_DENSE_WEIGHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def synchronize_dflash2_mlp_scales(
    payload: torch.Tensor, scales: torch.Tensor, tp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put TP partials in one scale domain before row-parallel reduction."""
    if tp_size <= 1:
        return payload, scales
    gathered_scales = tensor_model_parallel_all_gather(scales, dim=0)
    shared_scales = gathered_scales.view(tp_size, -1).amax(dim=0)
    scale_ratio = (scales / shared_scales).to(payload.dtype).unsqueeze(-1)
    return payload * scale_ratio, shared_scales


def should_enable_dflash2_sm75(config: object, runtime_dtype: torch.dtype) -> bool:
    """Select the codec only for the BF16 DFlash2 checkpoint on Turing."""
    if os.getenv("VLLM_DFLASH2_SM75_TRANSPORT", "1") == "0":
        return False
    if runtime_dtype is not torch.float16 or not torch.cuda.is_available():
        return False

    checkpoint_dtype = getattr(config, "torch_dtype", None)
    if checkpoint_dtype is None:
        checkpoint_dtype = getattr(config, "dtype", None)
    is_bf16_checkpoint = checkpoint_dtype is torch.bfloat16 or str(
        checkpoint_dtype
    ).lower() in {"bf16", "bfloat16", "torch.bfloat16"}
    return is_bf16_checkpoint and torch.cuda.get_device_capability() == (7, 5)


def _round_to_bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


@triton.jit
def _bf16_rne(value):
    """Keep BF16 precision in an FP32 Triton register."""
    word = value.to(tl.uint32, bitcast=True)
    tie = (word >> 16) & 1
    rounded_word = (word + 0x7FFF + tie) & 0xFFFF0000
    return rounded_word.to(tl.float32, bitcast=True)


@triton.jit
def _sm75_encode_mlp_rows(
    source_ptr,
    payload_ptr,
    scale_ptr,
    down_projection_l2_bound,
    intermediate_width: tl.constexpr,
    activation_gain: tl.constexpr,
    payload_limit: tl.constexpr,
    accumulation_limit: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    valid = lane < intermediate_width
    input_base = token * (2 * intermediate_width) + lane

    gate = tl.load(source_ptr + input_base, mask=valid, other=0.0).to(tl.float32)
    value = tl.load(
        source_ptr + input_base + intermediate_width, mask=valid, other=0.0
    ).to(tl.float32)
    gate = _bf16_rne(gate * activation_gain)
    value = _bf16_rne(value * activation_gain)
    gated = _bf16_rne(gate * tl.sigmoid(gate))
    product = _bf16_rne(gated * value)

    largest = tl.max(tl.abs(product), axis=0)
    product_l2 = tl.sqrt(tl.sum(product * product, axis=0))
    required_scale = tl.maximum(
        largest / payload_limit,
        product_l2 * down_projection_l2_bound / accumulation_limit,
    )
    exponent = tl.ceil(tl.log2(tl.maximum(required_scale, 1.0)))
    row_scale = tl.exp2(exponent)
    tl.store(scale_ptr + token, row_scale)
    tl.store(
        payload_ptr + token * intermediate_width + lane,
        product / row_scale,
        mask=valid,
    )


@triton.jit
def _sm75_restore_mlp_rows(
    payload_ptr,
    scale_ptr,
    restored_ptr,
    row_width: tl.constexpr,
    residual_gain: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    valid = lane < row_width
    offsets = token * row_width + lane
    payload = tl.load(payload_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    multiplier = tl.load(scale_ptr + token).to(tl.float32) / residual_gain
    tl.store(restored_ptr + offsets, payload * multiplier, mask=valid)


@triton.jit
def _sm75_residual_norm(
    output_ptr,
    state_ptr,
    input_ptr,
    residual_ptr,
    weight_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    residual_gain: tl.constexpr,
    WRITES_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    valid = lane < hidden_size
    offsets = token * hidden_size + lane
    state = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    if WRITES_STATE:
        residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(
            tl.float32
        )
        state = _bf16_rne(state * residual_gain + residual)
        tl.store(state_ptr + offsets, state, mask=valid)

    mean_square = tl.sum(state * state, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(mean_square + epsilon)
    weight = tl.load(weight_ptr + lane, mask=valid, other=0.0).to(tl.float32)
    normalized = _bf16_rne(state * inverse_rms * _bf16_rne(weight))
    tl.store(output_ptr + offsets, normalized, mask=valid)


def _block_size(width: int) -> int:
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError(f"DFlash2 SM75 row width {width} exceeds codec limits.")
    return block


def _num_warps(width: int) -> int:
    if width <= 1024:
        return 4
    if width <= 8192:
        return 8
    return 16


def _encode_reference(
    gate_up: torch.Tensor,
    down_projection_l2_bound: float,
    accumulation_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, value = gate_up.float().chunk(2, dim=-1)
    gate = _round_to_bf16(gate * _ACTIVATION_GAIN)
    value = _round_to_bf16(value * _ACTIVATION_GAIN)
    product = _round_to_bf16(_round_to_bf16(gate * torch.sigmoid(gate)) * value)
    product_l2 = torch.linalg.vector_norm(product, dim=-1)
    required_scale = torch.maximum(
        product.abs().amax(dim=-1) / _HALF_PAYLOAD_LIMIT,
        product_l2 * down_projection_l2_bound / accumulation_limit,
    )
    scale = torch.exp2(
        torch.ceil(
            torch.log2(torch.maximum(required_scale, torch.ones_like(required_scale)))
        )
    )
    return (product / scale.unsqueeze(-1)).to(gate_up.dtype), scale


def encode_dflash2_mlp_sm75(
    gate_up: torch.Tensor,
    *,
    down_projection_l2_bound: float = 0.0,
    accumulation_limit: float = _HALF_PAYLOAD_LIMIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode BF16 SwiGLU rows without overflowing a FP16 down projection.

    ``down_projection_l2_bound`` is the maximum L2 norm of one local output
    row in the following row-parallel projection.  Cauchy-Schwarz makes the
    resulting scale a conservative bound on every local partial sum.
    """
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2:
        raise ValueError(
            "DFlash2 SM75 MLP input must be [..., 2 * intermediate_size], "
            f"got {tuple(gate_up.shape)}."
        )
    if down_projection_l2_bound < 0 or accumulation_limit <= 0:
        raise ValueError("DFlash2 SM75 projection bounds must be positive.")
    if not gate_up.is_cuda:
        return _encode_reference(
            gate_up, down_projection_l2_bound, accumulation_limit
        )
    if gate_up.dtype is not torch.float16:
        raise ValueError("DFlash2 SM75 MLP codec requires FP16 CUDA inputs.")

    source = gate_up.contiguous()
    width = source.shape[-1] // 2
    rows = source.numel() // (2 * width)
    payload = torch.empty(
        (*source.shape[:-1], width), dtype=source.dtype, device=source.device
    )
    scales = torch.empty(rows, dtype=torch.float32, device=source.device)
    _sm75_encode_mlp_rows[(rows,)](
        source,
        payload,
        scales,
        down_projection_l2_bound,
        intermediate_width=width,
        activation_gain=_ACTIVATION_GAIN,
        payload_limit=_HALF_PAYLOAD_LIMIT,
        accumulation_limit=accumulation_limit,
        BLOCK=_block_size(width),
        num_warps=_num_warps(width),
    )
    return payload, scales


def restore_dflash2_mlp_sm75(
    projected: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Decode an MLP projection into the wide residual representation."""
    if projected.ndim < 2:
        raise ValueError("DFlash2 SM75 projected MLP output must have rows.")
    width = projected.shape[-1]
    rows = projected.numel() // width
    if scales.numel() != rows:
        raise ValueError("DFlash2 SM75 MLP scale count does not match its rows.")
    if not projected.is_cuda:
        return (
            projected.float() * (scales.reshape(-1, 1) / _RESIDUAL_GAIN)
        ).to(projected.dtype)
    if projected.dtype is not torch.float16:
        raise ValueError("DFlash2 SM75 MLP restore requires FP16 CUDA inputs.")

    source = projected.contiguous()
    restored = torch.empty_like(source)
    _sm75_restore_mlp_rows[(rows,)](
        source,
        scales,
        restored,
        row_width=width,
        residual_gain=_RESIDUAL_GAIN,
        BLOCK=_block_size(width),
        num_warps=_num_warps(width),
    )
    return restored


class Sm75DFlash2ResidualNorm(nn.Module):
    """RMSNorm that bridges FP16 layers and DFlash2's wide residual state."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype), requires_grad=False
        )

    def _run_reference(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state = x.float()
        if residual is not None:
            state = _round_to_bf16(state * _RESIDUAL_GAIN + residual.float())
        output = _round_to_bf16(
            state
            * torch.rsqrt(
                state.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
            )
            * _round_to_bf16(self.weight.float())
        ).to(self.weight.dtype)
        return output if residual is None else (output, state)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            return self._run_reference(x, residual)
        if x.dtype not in (torch.float16, torch.float32):
            raise ValueError("DFlash2 SM75 RMSNorm requires FP16 or FP32 CUDA inputs.")

        source = x.contiguous()
        incoming_state = residual.contiguous() if residual is not None else source
        rows = source.numel() // self.hidden_size
        output = torch.empty_like(source, dtype=self.weight.dtype)
        next_state = (
            torch.empty_like(source, dtype=torch.float32)
            if residual is not None
            else output
        )
        _sm75_residual_norm[(rows,)](
            output,
            next_state,
            source,
            incoming_state,
            self.weight,
            hidden_size=self.hidden_size,
            epsilon=self.variance_epsilon,
            residual_gain=_RESIDUAL_GAIN,
            WRITES_STATE=residual is not None,
            BLOCK=_block_size(self.hidden_size),
            num_warps=_num_warps(self.hidden_size),
        )
        return output if residual is None else (output, next_state)


def _dense_float_weight(layer: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Return a layer's *effective* weight in float32.

    The SM75 transport bound must reflect the weights the kernel actually
    applies. Quantized checkpoints store low-precision values plus separate
    scales, so normalizing the serialized values directly overestimates the
    row norms by roughly ``1 / scale``. For block-FP8 drafts whose
    ``weight_scale_inv`` sits in ``[1e-3, 2e-2]`` that is a factor of 50-1000,
    which makes the transport pick an activation scale two orders of magnitude
    larger than needed: the FP16 headroom the bound exists to protect is
    wasted, and small activations are pushed towards the subnormal range.
    """
    # ``float8`` dtypes report ``is_floating_point == True`` even though they
    # are 8-bit payloads with separate scales, so the dense dtypes are listed
    # explicitly in _DENSE_WEIGHT_DTYPES.
    if weight.dtype in _DENSE_WEIGHT_DTYPES:
        return weight.float()

    scale = getattr(layer, "weight_scale_inv", None)
    if not isinstance(scale, torch.Tensor):
        scale = getattr(layer, "weight_scale", None)
    if not isinstance(scale, torch.Tensor):
        raise ValueError(
            "DFlash2 SM75 transport cannot derive the down-projection bound: "
            f"weight dtype is {weight.dtype} and the layer exposes no "
            "weight_scale_inv/weight_scale to dequantize it. A packed weight "
            "means the bound was requested after the quantization kernel "
            "rewrote the parameter; initialize it during load_weights instead."
        )

    flat_weight = weight.float()
    flat_scale = scale.float()
    if flat_scale.numel() == 1:
        return flat_weight * flat_scale.reshape(())

    if flat_scale.dim() == 2 and flat_scale.shape[0] == 1:
        if flat_scale.shape[1] != weight.shape[1]:
            raise ValueError(
                "DFlash2 SM75 transport cannot align per-channel scales "
                f"{tuple(scale.shape)} with the local down-projection weight "
                f"{tuple(weight.shape)}."
            )
        return flat_weight * flat_scale.expand_as(flat_weight)

    if flat_scale.dim() == 2:
        block = getattr(layer, "weight_block_size", None) or (128, 128)
        block_n, block_k = int(block[0]), int(block[1])
        expanded = flat_scale.repeat_interleave(block_n, 0).repeat_interleave(
            block_k, 1
        )
        expanded = expanded[: weight.shape[0], : weight.shape[1]]
        if expanded.shape == weight.shape:
            return flat_weight * expanded

    raise ValueError(
        "DFlash2 SM75 transport cannot align block scales "
        f"{tuple(scale.shape)} with the local down-projection weight "
        f"{tuple(weight.shape)}."
    )


class Sm75DFlash2MLP(nn.Module):
    """SM75 implementation of DFlash2's BF16-trained SwiGLU transport."""

    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj
        self._down_projection_l2_bound: float | None = None
        self._accumulation_limit: float | None = None

    def initialize_down_projection_bound(self) -> None:
        """Derive a finite FP16 partial-sum bound from loaded local weights."""
        weight = getattr(self.down_proj, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(
                "DFlash2 SM75 transport requires a 2-D down projection weight."
            )
        tp_size = max(1, int(getattr(self.down_proj, "tp_size", 1)))
        with torch.no_grad():
            dense_weight = _dense_float_weight(self.down_proj, weight)
            self._down_projection_l2_bound = float(
                torch.linalg.vector_norm(dense_weight, dim=1).amax().item()
            )
        # Each rank contributes a FP16 tensor to the row-parallel all-reduce.
        self._accumulation_limit = _HALF_PAYLOAD_LIMIT / tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._down_projection_l2_bound is None
            or self._accumulation_limit is None
        ):
            raise RuntimeError(
                "DFlash2 SM75 projection bounds were not initialized after "
                "loading weights."
            )
        gate_up, _ = self.gate_up_proj(x / _ACTIVATION_GAIN)
        payload, scales = encode_dflash2_mlp_sm75(
            gate_up,
            down_projection_l2_bound=self._down_projection_l2_bound,
            accumulation_limit=self._accumulation_limit,
        )
        payload, scales = synchronize_dflash2_mlp_scales(
            payload, scales, int(getattr(self.down_proj, "tp_size", 1))
        )
        projected, _ = self.down_proj(payload)
        return restore_dflash2_mlp_sm75(projected, scales)


def dflash2_sm75_residual_gain() -> float:
    return _RESIDUAL_GAIN
