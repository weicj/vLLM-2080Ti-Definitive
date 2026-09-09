# SPDX-License-Identifier: Apache-2.0
"""Physical KV-cache layout descriptors.

This small module mirrors the layout enum used by newer vLLM model
implementations while keeping the 0.2.x cache planner independent from the
full cache interface.
"""

from enum import Enum

_DIM_L, _DIM_B, _DIM_H, _DIM_N, _DIM_C = 0, 1, 2, 3, 4


class KVCacheLayout(Enum):
    """Stride permutation for logical ``[L, B, H, N, C]`` cache storage."""

    LBHNC = (0, 1, 2, 3, 4)
    LBNHC = (0, 1, 3, 2, 4)
    LHBNC = (0, 2, 1, 3, 4)
    BLHNC = (1, 0, 2, 3, 4)
    BLNHC = (1, 0, 3, 2, 4)
    BHLNC = (1, 2, 0, 3, 4)

    @property
    def stride_order(self) -> tuple[int, ...]:
        return self.value

    @property
    def layer_view_order(self) -> tuple[int, ...]:
        return tuple(i - 1 for i in self.value if i != _DIM_L)

    @property
    def is_layer_compact(self) -> bool:
        return self.value[_DIM_L] == 0

    @property
    def is_block_contiguous(self) -> bool:
        return self.value[-3:] == (_DIM_H, _DIM_N, _DIM_C)

    @property
    def is_block_compact(self) -> bool:
        return set(self.value[:2]) == {_DIM_L, _DIM_B}

    @property
    def is_block_outermost(self) -> bool:
        return self.value[0] == _DIM_B


__all__ = ["KVCacheLayout"]
