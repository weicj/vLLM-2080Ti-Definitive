"""Pre-boot sanity for padded EXL3 linears: the padded output columns of the vision fc1 are
exactly zero (svh = 0 there), fc2 accepts zero-extended inputs, and the patched plugin exposes
the padding helpers and codebook-flag plumbing."""
import glob
import json
import struct
import sys

import torch
from safetensors import safe_open

import exllamav3_ext  # noqa: F401
from vllm_exl3 import exl3 as X

pack = sys.argv[1]
ok = True
names = {}
for fn in sorted(glob.glob(pack + "/model-*.safetensors")):
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for k in h:
        if k.startswith("model.visual.blocks.0.mlp."):
            names[k] = fn


def get(k):
    with safe_open(names[k], "pt", device="cuda") as f:
        return f.get_tensor(k)


p = "model.visual.blocks.0.mlp."
fc1 = X.make_linear_exl3(get(p + "linear_fc1.trellis"), get(p + "linear_fc1.suh"), get(p + "linear_fc1.svh"), None, get(p + "linear_fc1.mul1").reshape(1))
fc2 = X.make_linear_exl3(get(p + "linear_fc2.trellis"), get(p + "linear_fc2.suh"), get(p + "linear_fc2.svh"), None, get(p + "linear_fc2.mul1").reshape(1))
print("fc1 in/out:", fc1.in_features, fc1.out_features, "| fc2 in/out:", fc2.in_features, fc2.out_features)
x = torch.randn(8, 1152, device="cuda", dtype=torch.float16)
y1 = fc1.forward(x, {}, out_dtype=torch.float32)
tail_max = y1[:, 4304:].abs().max().item()
print("fc1 out", tuple(y1.shape), "padded columns max |y| =", tail_max, "| real columns mean |y| = %.4f" % y1[:, :4304].abs().mean().item())
ok &= tuple(y1.shape) == (8, 4352) and tail_max == 0.0 and bool(y1.isfinite().all())
h = torch.nn.functional.gelu(y1[:, :4304]).to(torch.float16)
h_pad = torch.nn.functional.pad(h, (0, 48))
y2 = fc2.forward(h_pad.contiguous(), {}, out_dtype=torch.float32)
print("fc2 out", tuple(y2.shape), "finite:", bool(y2.isfinite().all()), "mean |y| = %.4f" % y2.abs().mean().item())
ok &= tuple(y2.shape) == (8, 1152) and bool(y2.isfinite().all())
# padded input rows must not matter: garbage in the padded input columns must change nothing
h_junk = h_pad.clone()
h_junk[:, 4304:] = 1.2345
y2b = fc2.forward(h_junk.contiguous(), {}, out_dtype=torch.float32)
padding_independent = torch.equal(y2, y2b)
print("fc2 identical with junk padding rows:", padding_independent)
print("plugin helpers:", hasattr(X, "_exl3_pad128") and X._exl3_pad128(4304) == 4352, "| flags plumbing:", "_exl3_codebook_flags" in open(X.__file__).read())
ok &= padding_independent
ok &= hasattr(X, "_exl3_pad128") and "_exl3_codebook_flags" in open(X.__file__).read()
print("PAD_CHECK:", "PASS" if ok else "FAIL")
