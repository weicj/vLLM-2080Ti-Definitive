# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import ScalarType

# Weight widths whose int32 packing can be decoded here.
_PACKED_BOUND_BITS = (2, 4, 8)


@dataclass
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    out_type: torch.dtype | None = None


def _packed_layout(param: torch.Tensor) -> tuple[int, int, int]:
    """Return the ``(output_dim, input_dim, packed_dim)`` of a packed weight."""
    output_dim = getattr(param, "output_dim", None)
    input_dim = getattr(param, "input_dim", None)
    packed_dim = getattr(param, "packed_dim", None)
    if output_dim is None or input_dim is None or packed_dim is None:
        raise ValueError(
            "Packed weight layout metadata is missing; compute this bound "
            "before process_weights_after_loading() repacks the parameters."
        )
    return int(output_dim), int(input_dim), int(packed_dim)


def _unpack_int_codes(
    words: torch.Tensor, bits: int, num_inputs: int
) -> torch.Tensor:
    """Decode ``[outputs, words]`` int32 weights into FP32 integer codes."""
    if words.dtype is not torch.int32:
        raise ValueError(f"Packed weight must be int32, got {words.dtype}.")
    per_word = 32 // bits
    if words.shape[1] * per_word < num_inputs:
        raise ValueError(
            f"Packed weight holds {words.shape[1] * per_word} codes but "
            f"{num_inputs} inputs are expected."
        )
    shifts = torch.arange(0, 32, bits, device=words.device, dtype=torch.int32)
    mask = (1 << bits) - 1
    # Masking after the shift keeps sign-extended int32 words correct.
    codes = (words.unsqueeze(-1) >> shifts) & mask
    codes = codes.reshape(words.shape[0], words.shape[1] * per_word)
    return codes[:, :num_inputs].to(torch.float32)


def _scale_matrix(
    w_s: torch.Tensor, outputs: int, num_groups: int
) -> torch.Tensor:
    """Broadcast a weight scale to every ``(output, group)`` pair."""
    output_dim = getattr(w_s, "output_dim", None)
    scale = w_s.detach().to(torch.float32)
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(outputs, num_groups)
    if scale.dim() == 1:
        if scale.numel() != outputs:
            raise ValueError(
                f"Per-channel scale has {scale.numel()} entries, expected "
                f"{outputs}."
            )
        if num_groups != 1:
            raise ValueError("Per-channel scales need channelwise quantization.")
        return scale.reshape(outputs, 1)
    if scale.dim() != 2:
        raise ValueError(f"Unsupported weight scale rank {scale.dim()}.")
    if output_dim is None:
        if scale.shape[0] != outputs and scale.shape[1] == outputs:
            scale = scale.transpose(0, 1)
    elif output_dim == 1:
        scale = scale.transpose(0, 1)
    if scale.shape[0] != outputs:
        raise ValueError(
            f"Weight scale covers {scale.shape[0]} outputs, expected {outputs}."
        )
    if scale.shape[1] == 1:
        return scale.expand(outputs, num_groups)
    if scale.shape[1] != num_groups:
        raise ValueError(
            f"Weight scale covers {scale.shape[1]} groups, expected {num_groups}."
        )
    return scale


@dataclass(frozen=True)
class _PackedWeight:
    """A symmetric packed weight normalized to a ``[outputs, words]`` view."""

    codes: torch.Tensor
    bits: int
    num_inputs: int
    num_outputs: int
    num_groups: int
    group_width: int
    bias: float


def _packed_weight_geometry(
    config: MPLinearLayerConfig,
    w_q: torch.Tensor,
    w_zp: torch.Tensor | None,
) -> _PackedWeight:
    """Validate a symmetric packed weight and normalize its layout."""
    weight_type = config.weight_type
    if weight_type.is_floating_point():
        raise ValueError(
            f"Packed weight needs an integer weight type, got {weight_type}."
        )
    if config.zero_points:
        raise NotImplementedError(
            "Asymmetric quantized weights (zero points) cannot be decoded "
            "without their exact scale layout."
        )
    bits = weight_type.size_bits
    if bits not in _PACKED_BOUND_BITS:
        raise ValueError(f"Unsupported packed weight width {bits} bits.")

    num_inputs, num_outputs = config.partition_weight_shape
    group_width = num_inputs if config.group_size == -1 else config.group_size
    if group_width <= 0 or num_inputs % group_width:
        raise ValueError(
            f"Group size {config.group_size} does not divide {num_inputs} inputs."
        )

    if w_q.dim() != 2:
        raise ValueError(f"Packed weight must be 2-D, got {tuple(w_q.shape)}.")
    output_dim, input_dim, packed_dim = _packed_layout(w_q)
    if packed_dim != input_dim:
        raise ValueError("Packed axis must be the weight's input axis.")
    codes = w_q if output_dim == 0 else w_q.transpose(0, 1)
    if codes.shape[0] != num_outputs:
        raise ValueError(
            f"Packed weight has {codes.shape[0]} outputs, expected {num_outputs}."
        )
    return _PackedWeight(
        codes=codes,
        bits=bits,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_groups=num_inputs // group_width,
        group_width=group_width,
        bias=float(weight_type.bias),
    )


def packed_weight_column_norm_bound(
    config: MPLinearLayerConfig,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None = None,
    *,
    chunk_rows: int = 1024,
) -> float:
    """Largest effective weight L2 norm of one local output channel.

    ``W_eff[o, i] = (code[o, i] - bias) * scale[o, i // group_size]``, so the
    result upper-bounds ``max_o ||W_eff[o, :]||_2`` in FP32. Only symmetric
    weight-only quantization is decoded: an under-estimate would overflow an
    FP16 partial sum silently, so anything that cannot be decoded exactly is
    rejected instead of guessed.
    """
    packed = _packed_weight_geometry(config, w_q, w_zp)
    scales = _scale_matrix(w_s, packed.num_outputs, packed.num_groups)

    best = 0.0
    with torch.no_grad():
        for start in range(0, packed.num_outputs, chunk_rows):
            stop = min(start + chunk_rows, packed.num_outputs)
            codes = _unpack_int_codes(
                packed.codes[start:stop], packed.bits, packed.num_inputs
            )
            squares = (codes - packed.bias).square()
            per_group = squares.reshape(
                stop - start, packed.num_groups, packed.group_width
            )
            per_group = per_group.sum(-1)
            column = (per_group * scales[start:stop].square()).sum(-1).sqrt()
            best = max(best, float(column.amax()))
    return best


def packed_dequantized_weight(
    config: MPLinearLayerConfig,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None = None,
    *,
    start: int = 0,
    stop: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Materialize ``[start, stop)`` output rows of a symmetric packed weight.

    Consumers that need a dense copy of the local weight (e.g. a fused GEMM
    over several layers) get the same values the quantized kernel would
    dequantize internally, in ``dtype``.
    """
    packed = _packed_weight_geometry(config, w_q, w_zp)
    scales = _scale_matrix(w_s, packed.num_outputs, packed.num_groups)
    if stop is None:
        stop = packed.num_outputs
    stop = min(stop, packed.num_outputs)
    if not 0 <= start < stop:
        raise ValueError(f"Empty weight row range [{start}, {stop}).")

    with torch.no_grad():
        codes = _unpack_int_codes(
            packed.codes[start:stop], packed.bits, packed.num_inputs
        )
        values = (codes - packed.bias).reshape(
            stop - start, packed.num_groups, packed.group_width
        )
        values = values * scales[start:stop].unsqueeze(-1)
    return values.reshape(stop - start, packed.num_inputs).to(dtype)


class MPLinearKernel(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: str | None = None,
    ) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        if c.zero_points:
            assert w_zp_param_name is not None
        self.w_zp_name: str | None = w_zp_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _transform_param(
        self, layer: torch.nn.Module, name: str | None, fn: Callable
    ) -> None:
        if name is not None and getattr(layer, name, None) is not None:
            old_param = getattr(layer, name)
            new_param = fn(old_param)
            # replace the parameter with torch.nn.Parameter for TorchDynamo
            # compatibility
            replace_parameter(
                layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
            )

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,  # w_q
        torch.Tensor,  # w_s
        torch.Tensor | None,  # w_zp
    ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
        )

    def _require_packed_params(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return the checkpoint-layout packed weight, scale, and zero point."""
        act_type = self.config.act_type
        if act_type is not None and getattr(act_type, "itemsize", 2) == 1:
            raise NotImplementedError(
                "Activation-quantized weights break the FP16 activation "
                "transport this API exists for."
            )
        if getattr(layer, "g_idx", None) is not None:
            raise NotImplementedError(
                "Activation reordering (g_idx) does not expose a stable weight "
                "layout for dequantization."
            )
        w_q = getattr(layer, self.w_q_name, None)
        w_s = getattr(layer, self.w_s_name, None)
        if not isinstance(w_q, torch.Tensor) or not isinstance(w_s, torch.Tensor):
            raise ValueError(
                f"Layer {type(layer).__name__} is missing {self.w_q_name} or "
                f"{self.w_s_name}."
            )
        w_zp = None
        if self.config.zero_points and self.w_zp_name:
            w_zp = getattr(layer, self.w_zp_name, None)
        return w_q, w_s, w_zp

    def weight_column_norm_bound(self, layer: torch.nn.Module) -> float:
        """Upper bound on the effective local weight row L2 norm.

        Consumers that must bound a local GEMM partial sum in FP16 (the SM75
        DFlash2 transport) need the dequantized row norms of the local shard,
        which are not available from the unquantized weight accessor. This
        must run while the parameters are still in checkpoint layout, i.e.
        before ``process_weights_after_loading`` repacks them.
        """
        w_q, w_s, w_zp = self._require_packed_params(layer)
        return packed_weight_column_norm_bound(self.config, w_q, w_s, w_zp)

    def dequantized_weight(
        self,
        layer: torch.nn.Module,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Materialize output rows ``[start, stop)`` of the dequantized weight.

        Same checkpoint-layout contract as :meth:`weight_column_norm_bound`.
        """
        w_q, w_s, w_zp = self._require_packed_params(layer)
        return packed_dequantized_weight(
            self.config, w_q, w_s, w_zp, start=start, stop=stop, dtype=dtype
        )


def mp_linear_kernel(layer: torch.nn.Module) -> MPLinearKernel | None:
    """Return the mixed-precision kernel that owns ``layer``'s weight, if any.

    Quantization methods either own the kernel directly (AutoGPTQ stores it on
    the method) or delegate to a scheme kept on the layer (compressed-tensors
    stores ``layer.scheme``).
    """
    quant_method = getattr(layer, "quant_method", None)
    candidates = (
        getattr(quant_method, "kernel", None),
        getattr(quant_method, "scheme", None),
        getattr(layer, "scheme", None),
    )
    for candidate in candidates:
        if isinstance(candidate, MPLinearKernel):
            return candidate
        kernel = getattr(candidate, "kernel", None)
        if isinstance(kernel, MPLinearKernel):
            return kernel
    return None
