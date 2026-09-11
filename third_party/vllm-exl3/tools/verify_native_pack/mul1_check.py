"""Pre-boot sanity for mul1 packs through the plugin's LinearEXL3 path and MoE marker validator."""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import exllamav3_ext  # noqa: F401  (needs torch imported first)
from vllm_exl3 import exl3 as X

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} <pack_dir>")

pack = sys.argv[1]
P = "model.language_model.layers.0.mlp.shared_expert.gate_proj."


def load_linear_tensors(pack_dir: str):
    names = tuple(P + suffix for suffix in ("trellis", "suh", "svh", "mul1"))
    for shard in sorted(Path(pack_dir).glob("*.safetensors")):
        with safe_open(shard, "pt", device="cuda") as f:
            if names[0] in f.keys():
                return tuple(f.get_tensor(name) for name in names)
    raise FileNotFoundError(f"{names[0]} not found in {pack_dir}/*.safetensors")


tr, suh, svh, mul1 = load_linear_tensors(pack)
ok = True
print("marker", int(mul1), "== MUL1_MARKER:", int(mul1) == X.MUL1_MARKER_SIGNED_INT32, "trellis", tuple(tr.shape))
ok &= int(mul1) == X.MUL1_MARKER_SIGNED_INT32
marker = mul1.reshape(1)
lin = X.make_linear_exl3(tr, suh, svh, None, marker)
print("LinearEXL3 attrs: mcg=", getattr(lin, "mcg", "n/a"), "mul1=", getattr(lin, "mul1", "n/a"))
x = torch.randn(8, 2560, device="cuda", dtype=torch.float16)
y = lin.forward(x, {}, out_dtype=torch.float32)
lin2 = X.make_linear_exl3(
    tr, suh, svh, marker, None
)
y2 = lin2.forward(x, {}, out_dtype=torch.float32)
differs = not torch.allclose(y, y2)
print("mul1 forward finite:", y.isfinite().all().item(), "mean|y|=%.4f" % y.abs().mean().item(), "differs from MCG decode:", differs)
ok &= bool(y.isfinite().all().item()) and differs
m = torch.zeros(4, 2, 1, dtype=torch.int32)
u = torch.full((4, 2, 1), X.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)
X._check_moe_codebook_markers(m, u, "w13")
print("validator: all-mul1 ok")
try:
    X._check_moe_codebook_markers(m, torch.zeros_like(u), "w13")
    ok = False
    print("validator FAILED to reject missing markers")
except RuntimeError as e:
    print("validator rejects missing markers:", str(e)[:70])
print("MUL1_CHECK:", "PASS" if ok else "FAIL")
