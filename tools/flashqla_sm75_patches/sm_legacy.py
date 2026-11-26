# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None
_REQUIRED_EXTENSION_SYMBOLS = ("gdn_forward", "gdn_forward_varlen")


def _validate_extension(extension):
    """Reject an old SM75 binary before it can enter a varlen prefill."""
    missing = [
        name for name in _REQUIRED_EXTENSION_SYMBOLS if not hasattr(extension, name)
    ]
    if missing:
        raise RuntimeError(
            "FlashQLA SM70/SM75 extension is stale; missing exported symbols: "
            + ", ".join(missing)
            + ". Re-run build.sh after applying the packed-varlen patch."
        )
    return extension


def _debug_legacy_enabled() -> bool:
    return os.getenv("SGLANG_DEBUG_GDN_SM75", "").lower() in ("1", "true", "yes")


def _debug_sync(label: str, tensor: torch.Tensor | None = None) -> None:
    if not _debug_legacy_enabled() or not torch.cuda.is_available():
        return
    try:
        if tensor is not None and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()
        print(f"[FLASHQLA-LEGACY] sync ok: {label}", flush=True)
    except Exception as exc:
        print(f"[FLASHQLA-LEGACY] sync failed at {label}: {type(exc).__name__}: {exc}", flush=True)
        raise


def _debug_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if not _debug_legacy_enabled():
        return
    if tensor is None:
        print(f"[FLASHQLA-LEGACY] {name}: None", flush=True)
        return
    print(
        f"[FLASHQLA-LEGACY] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} stride={tensor.stride()} contiguous={tensor.is_contiguous()}",
        flush=True,
    )


def _clear_stale_extension_lock() -> None:
    default_root = Path(__file__).resolve().parents[4] / ".torch_extensions_vllm_flashqla_legacy"
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", str(default_root)))
    # PyTorch names this directory from the active Python/CUDA ABI (for
    # example py312_cu130); do not retain a lock path from an older build.
    for lock in root.glob("*/flash_qla_legacy_gdn/lock"):
        so = lock.with_name("flash_qla_legacy_gdn.so")
        try:
            if not lock.exists() or not so.exists() or lock.stat().st_size != 0:
                continue
            age = __import__("time").time() - lock.stat().st_mtime
            if age > 30:
                lock.rename(lock.with_name(f"lock.stale-{int(__import__('time').time())}"))
                if _debug_legacy_enabled():
                    print(f"[FLASHQLA-LEGACY] cleared stale extension lock age={age:.1f}s", flush=True)
        except FileNotFoundError:
            continue


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    if not torch.cuda.is_available():
        raise RuntimeError("SM70/SM75 legacy GDN backend requires CUDA")

    # The Docker runtime ships the extension built against its Torch/CUDA ABI.
    try:
        import flash_qla_legacy_gdn
    except ModuleNotFoundError as exc:
        if exc.name != "flash_qla_legacy_gdn":
            raise
    else:
        _EXT = _validate_extension(flash_qla_legacy_gdn)
        return _EXT

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    _clear_stale_extension_lock()
    src = Path(__file__).with_name("csrc") / "gdn_forward.cu"
    default_root = Path(__file__).resolve().parents[4] / ".torch_extensions_vllm_flashqla_legacy"
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(default_root))
    extension = load(
        name="flash_qla_legacy_gdn",
        sources=[str(src)],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=bool(int(os.environ.get("FLASH_QLA_LEGACY_VERBOSE_BUILD", "0"))),
    )
    _EXT = _validate_extension(extension)
    return _EXT


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    tensors = [q, k, v, g, beta]
    if initial_state is not None:
        tensors.append(initial_state)

    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("legacy GDN tensors must be CUDA tensors")
    if any(tensor.dtype not in (torch.float16, torch.float32) for tensor in tensors):
        raise ValueError("legacy GDN backend currently supports float16/float32 tensors only")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("legacy GDN q/k/v tensors must have the same dtype")
    if beta.dtype != g.dtype:
        raise ValueError("legacy GDN g/beta tensors must have the same dtype")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D]")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")

    batch, tokens, q_heads, dim = q.shape
    if v.shape[0] != batch or v.shape[1] != tokens or v.shape[3] != dim:
        raise ValueError("v must have shape [B, T, Hv, D] matching q/k")
    if g.shape != beta.shape or g.shape != v.shape[:3]:
        raise ValueError("g and beta must have shape [B, T, Hv]")
    if v.shape[2] % q_heads != 0:
        raise ValueError("Hv must be divisible by Hq")
    if dim not in (16, 32, 64, 128):
        raise ValueError("legacy GDN backend supports D in {16, 32, 64, 128}")
    if initial_state is not None and initial_state.shape != (batch, v.shape[2], dim, dim):
        raise ValueError("initial_state must have shape [B, Hv, D, D]")


def chunk_gated_delta_rule_fwd_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the experimental SM70/SM75 forward-only GDN backend.

    This legacy backend is intentionally explicit. It does not replace the
    Hopper/SM90 TileLang path and currently supports contiguous float16/float32
    tensors for inference-oriented forward execution.

    Shapes:
        q, k: [B, T, Hq, D]
        v: [B, T, Hv, D]
        g, beta: [B, T, Hv]
        initial_state: optional [B, Hv, D, D]

    Returns:
        output: [B, T, Hv, D]
        final_state: [B, Hv, D, D]
    """

    _debug_tensor("q", q)
    _debug_tensor("k", k)
    _debug_tensor("v", v)
    _debug_tensor("g", g)
    _debug_tensor("beta", beta)
    _debug_tensor("initial_state", initial_state)
    _debug_sync("before input check", q)
    _check_inputs(q, k, v, g, beta, initial_state)
    _debug_sync("after input check", q)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    ext = _load_ext()
    _debug_sync("after load_ext", q)
    out, final_state = ext.gdn_forward(q, k, v, g, beta, initial_state, float(scale))
    _debug_tensor("out", out)
    _debug_tensor("final_state", final_state)
    _debug_sync("after ext.gdn_forward", out)
    return out, final_state


def chunk_gated_delta_rule_fwd_legacy_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one packed-varlen GDN batch in a single CUDA launch.

    Inputs use vLLM's packed layout ``[1, total_tokens, heads, dim]`` and
    ``cu_seqlens`` identifies each request. The CUDA grid maps its batch axis
    to requests, so their recurrent scans execute concurrently.
    """
    _check_inputs(q, k, v, g, beta, None)
    if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32 on the same CUDA device")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [batch + 1]")
    batch = cu_seqlens.numel() - 1
    expected_state = (batch, v.shape[2], q.shape[3], q.shape[3])
    if initial_state is None or initial_state.shape != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")
    if not initial_state.is_cuda or initial_state.dtype not in (
        torch.float16,
        torch.float32,
    ):
        raise ValueError("initial_state must be a CUDA float16/float32 tensor")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ext = _load_ext()
    return ext.gdn_forward_varlen(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        float(scale),
        cu_seqlens,
    )
