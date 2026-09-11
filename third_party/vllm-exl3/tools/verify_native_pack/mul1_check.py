"""Pre-boot sanity for mul1 packs through the plugin's LinearEXL3 path and MoE marker validator."""
import sys

import torch
from safetensors import safe_open

import exllamav3_ext  # noqa: F401  (needs torch imported first)
from vllm_exl3 import exl3 as X

pack = sys.argv[1]
P = "model.language_model.layers.0.mlp.shared_expert.gate_proj."
with safe_open(pack + "/model-00001-of-00007.safetensors", "pt", device="cuda") as f:
    tr, suh, svh, mul1 = (f.get_tensor(P + s) for s in ("trellis", "suh", "svh", "mul1"))
ok = True
print("marker", int(mul1), "== MUL1_MARKER:", int(mul1) == X.MUL1_MARKER_SIGNED_INT32, "trellis", tuple(tr.shape))
ok &= int(mul1) == X.MUL1_MARKER_SIGNED_INT32
lin = X.make_linear_exl3(tr, suh, svh, None, mul1.reshape(1))
print("LinearEXL3 attrs: mcg=", getattr(lin, "mcg", "n/a"), "mul1=", getattr(lin, "mul1", "n/a"))
x = torch.randn(8, 2560, device="cuda", dtype=torch.float16)
y = lin.forward(x, {}, out_dtype=torch.float32)
lin2 = X.make_linear_exl3(
    tr, suh, svh, torch.tensor([X.MCG_MARKER_SIGNED_INT32], dtype=torch.int32, device="cuda"), None
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
