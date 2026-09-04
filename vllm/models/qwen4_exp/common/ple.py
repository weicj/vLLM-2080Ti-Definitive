# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp PLE helpers."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PLEShardOverlap:
    """Source and destination slices for one checkpoint embedding shard."""

    source_start: int
    destination_start: int
    row_count: int


def compute_ple_shard_overlap(
    *,
    checkpoint_start: int,
    checkpoint_rows: int,
    tp_start: int,
    tp_end: int,
) -> PLEShardOverlap | None:
    """Compute the overlap of a checkpoint shard and one TP vocabulary range."""

    if checkpoint_start < 0 or checkpoint_rows < 0:
        raise ValueError("checkpoint shard bounds must be non-negative")
    if tp_start < 0 or tp_end < tp_start:
        raise ValueError("invalid TP vocabulary range")
    checkpoint_end = checkpoint_start + checkpoint_rows
    overlap_start = max(checkpoint_start, tp_start)
    overlap_end = min(checkpoint_end, tp_end)
    if overlap_start >= overlap_end:
        return None
    return PLEShardOverlap(
        source_start=overlap_start - checkpoint_start,
        destination_start=overlap_start - tp_start,
        row_count=overlap_end - overlap_start,
    )


def copy_ple_embedding_shard_(
    destination: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    checkpoint_start: int,
    tp_start: int,
    tp_end: int,
) -> int:
    """Copy the overlapping rows of a PLE checkpoint shard into a TP table."""

    if destination.ndim == 0 or loaded_weight.ndim != destination.ndim:
        raise ValueError("destination and loaded weight must have matching ranks")
    if destination.shape[1:] != loaded_weight.shape[1:]:
        raise ValueError(
            "embedding shard dimensions do not match: "
            f"{tuple(destination.shape[1:])} != {tuple(loaded_weight.shape[1:])}"
        )
    if destination.shape[0] < tp_end - tp_start:
        raise ValueError("destination does not cover the requested TP range")
    overlap = compute_ple_shard_overlap(
        checkpoint_start=checkpoint_start,
        checkpoint_rows=loaded_weight.shape[0],
        tp_start=tp_start,
        tp_end=tp_end,
    )
    if overlap is None:
        return 0
    source = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
    target = destination.narrow(0, overlap.destination_start, overlap.row_count)
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return overlap.row_count


class DiskMappedPLEEmbedding(nn.Module):
    """Read a split PLE embedding directly from safetensors mappings.

    The Qwen4-Exp PLE table is tens of gigabytes even in FP8 form and cannot
    be materialized as a regular ``torch.nn.Parameter`` on hosts with modest
    system RAM. ``safe_open().get_tensor()`` keeps each source tensor backed
    by the file mapping, so only rows touched by a request become resident.
    This class is intentionally CPU-only and is used by the PLE offload
    process.
    """

    is_disk_mapped = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        key_prefix: str,
        org_vocab_size: int,
        embedding_dim: int,
        split_ngram_parts: int,
        scale_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if org_vocab_size <= 0 or embedding_dim <= 0 or split_ngram_parts <= 0:
            raise ValueError("invalid disk-mapped PLE embedding dimensions")
        root = Path(checkpoint_path)
        if not root.is_dir():
            raise FileNotFoundError(f"PLE checkpoint directory not found: {root}")
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                "disk-mapped PLE requires a safetensors index: " f"{index_path}"
            )

        # Import lazily so normal GPU workers do not open checkpoint files.
        from safetensors import safe_open

        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors weight_map in {index_path}")

        self.org_vocab_size = int(org_vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = self.org_vocab_size
        self.org_vocab_size_padded = self.org_vocab_size
        self.num_embeddings_padded = self.org_vocab_size
        self.num_added_embeddings = 0
        self.weight_dtype = torch.float8_e4m3fn
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=self.org_vocab_size,
        )
        # Keep the scale as a regular small parameter so the normal loader can
        # still populate ``ngram_embedding.weight_scale``.
        # Keep this scale in the runtime dtype: a BF16 checkpoint metadata
        # tensor is not compilable in a Turing CUDA graph.
        self.weight_scale = nn.Parameter(
            torch.ones(1, dtype=scale_dtype), requires_grad=False
        )

        shard_size = (self.org_vocab_size + split_ngram_parts - 1) // split_ngram_parts
        self._shard_size = shard_size
        self._shards: dict[int, torch.Tensor] = {}
        self._handles: dict[str, Any] = {}
        # Conditional-generation wrappers expose the language model as
        # ``language_model.model`` in vLLM, while the Hugging Face checkpoint
        # stores the same subtree as ``model.language_model``.  The normal
        # weight mapper handles this during model loading, but disk-mapped PLE
        # resolves names directly from the safetensors index.  Keep the vLLM
        # spelling first, then try the checkpoint spelling and the older
        # unwrapped alias used by early Qwen4-Exp experiments.
        key_prefixes = [key_prefix]
        if "language_model.model." in key_prefix:
            key_prefixes.append(
                key_prefix.replace("language_model.model.", "model.language_model.", 1)
            )
        if ".model." in key_prefix:
            key_prefixes.append(key_prefix.replace(".model.", ".", 1))

        # Build the suffix index once. The checkpoint has many non-PLE tensor
        # names, and scanning that complete map for each split embedding shard
        # makes worker startup quadratic in practice.
        shard_matches: dict[int, list[str]] = {}
        markers = tuple(
            f"{prefix}.ngram_embedding.shard_" for prefix in key_prefixes
        )
        for tensor_name in weight_map:
            if not tensor_name.endswith(".weight"):
                continue
            for marker in markers:
                marker_index = tensor_name.rfind(marker)
                if marker_index < 0:
                    continue
                shard_text = tensor_name[marker_index + len(marker) : -len(".weight")]
                if shard_text.isdigit():
                    shard_matches.setdefault(int(shard_text), []).append(tensor_name)
                break

        for shard_index in range(split_ngram_parts):
            suffixes = [
                f"{prefix}.ngram_embedding.shard_{shard_index}.weight"
                for prefix in key_prefixes
            ]
            matching = shard_matches.get(shard_index, [])
            if len(matching) != 1:
                raise RuntimeError(
                    "unable to resolve exactly one PLE shard in checkpoint: "
                    f"suffixes={suffixes!r}, matches={matching}"
                )
            tensor_name = matching[0]
            file_name = weight_map[tensor_name]
            file_path = str(root / file_name)
            handle = self._handles.get(file_path)
            if handle is None:
                handle = safe_open(file_path, framework="pt", device="cpu")
                self._handles[file_path] = handle
            tensor = handle.get_tensor(tensor_name)
            expected_rows = min(
                shard_size, self.org_vocab_size - shard_index * shard_size
            )
            if tuple(tensor.shape) != (expected_rows, self.embedding_dim):
                raise ValueError(
                    "PLE shard shape mismatch: "
                    f"{tensor_name} expected {(expected_rows, self.embedding_dim)}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.dtype != self.weight_dtype:
                raise ValueError(
                    f"disk-mapped PLE expects FP8 shards, got {tensor.dtype} "
                    f"for {tensor_name}"
                )
            self._shards[shard_index] = tensor

    def lookup(
        self,
        indices: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather FP8 rows without constructing the full embedding table."""
        if indices.device.type != "cpu":
            raise RuntimeError("disk-mapped PLE lookup requires CPU indices")
        flat_indices = indices.reshape(-1).to(dtype=torch.long)
        if flat_indices.numel() and (
            int(flat_indices.min()) < 0
            or int(flat_indices.max()) >= self.org_vocab_size
        ):
            raise IndexError("PLE embedding index is outside the mapped table")
        expected_shape = (flat_indices.numel(), self.embedding_dim)
        if out is None:
            output = torch.empty(expected_shape, dtype=self.weight_dtype)
        else:
            if tuple(out.shape) != expected_shape or out.device.type != "cpu":
                raise ValueError(
                    f"invalid PLE lookup output shape/device: {tuple(out.shape)} "
                    f"on {out.device}, expected {expected_shape} on CPU"
                )
            output = out

        if flat_indices.numel() == 0:
            return output
        shard_indices = torch.div(flat_indices, self._shard_size, rounding_mode="floor")
        for shard_index in torch.unique(shard_indices, sorted=True).tolist():
            mask = shard_indices == shard_index
            local_indices = flat_indices[mask] - shard_index * self._shard_size
            rows = self._shards[int(shard_index)].index_select(0, local_indices)
            output[mask] = rows
        return output

    def close(self) -> None:
        """Release safetensors mappings before the offload process exits."""
        self._shards.clear()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DiskMappedPLEEmbedding",
    "PLEShardOverlap",
    "compute_ple_shard_overlap",
    "copy_ple_embedding_shard_",
]
