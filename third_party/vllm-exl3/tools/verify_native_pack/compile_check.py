"""Pre-boot check: the EXL3 custom ops register, run eagerly, and trace under
torch.compile(fullgraph=True) for both a padded dense linear and the n-gram lookup."""
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
print("ops ready:", X._EXL3_OPS_READY, "| ops present:", hasattr(torch.ops.vllm, "exl3_linear_forward"), hasattr(torch.ops.vllm, "exl3_ngram_lookup"))
ok &= X._EXL3_OPS_READY

# --- dense linear (vision fc1, padded 1152 -> 4304 stored as 4352) --------------------------
names = {}
for fn in sorted(glob.glob(pack + "/model-*.safetensors")):
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    for k in h:
        if k.startswith("model.visual.blocks.0.mlp.linear_fc1."):
            names[k] = fn


def get(k):
    with safe_open(names[k], "pt", device="cuda") as f:
        return f.get_tensor(k)


p = "model.visual.blocks.0.mlp.linear_fc1."
lin = X.make_linear_exl3(get(p + "trellis"), get(p + "suh"), get(p + "svh"), None, get(p + "mul1").reshape(1))
cfg = X.Exl3Config(bits=3, codebook="mul1", scope="native_all_linears")
layer = torch.nn.Module()
layer.prefix = "visual.blocks.0.mlp.linear_fc1"
layer.quant_method = X.Exl3LinearMethod(cfg, bits=5)
layer._exl3_linears = [lin]
layer._exl3_linear_n_shards = 1
layer._exl3_linear_output_partition_sizes = [4352]
layer._exl3_linear_true_out = [4304]
layer._exl3_linear_true_in = 1152
layer._exl3_linear_padded = True
layer._exl3_linear_input_size_per_partition = 1152
layer._exl3_linear_bf16_shards = []
layer._exl3_opaque_name = X._exl3_register_opaque_layer(layer, "linear")
print("linear op name:", layer._exl3_opaque_name)
x = torch.randn(6, 1152, device="cuda", dtype=torch.bfloat16)
bias = get(p + "bias")[:4304].to(torch.bfloat16)
y_eager = layer.quant_method._apply_impl(layer, x) + bias
y_op = layer.quant_method.apply(layer, x, bias)
print("linear eager vs op:", tuple(y_op.shape), y_op.dtype, "equal:", torch.equal(y_eager, y_op))
ok &= torch.equal(y_eager, y_op) and tuple(y_op.shape) == (6, 4304)


def f_lin(t):
    return layer.quant_method.apply(layer, t, bias)


y_c = torch.compile(f_lin, fullgraph=True)(x)
print("linear under torch.compile(fullgraph=True):", tuple(y_c.shape), "equal to eager:", torch.equal(y_c, y_eager))
ok &= torch.equal(y_c, y_eager)

# --- n-gram lookup on a small fake table ---------------------------------------------------
PFX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
with safe_open(pack + "/ngram_embedding.safetensors", "pt") as f:
    head_bias = f.get_tensor(PFX + "head_bias")
    rows = f.get_slice(PFX + "shard_0.trellis")[0:2048]
ncfg = X.Exl3Config(bits=3, codebook="mul1", scope="native_all_linears",
                    ngram_embedding={"bits": 5, "num_shards": 2, "rows_per_shard": 1024, "num_heads": 16, "modules": ["ngram_embedding"]})
spec = ncfg._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
m = X.Exl3EmbeddingMethod(ncfg, spec)
emb = torch.nn.Module()
emb.tp_rank, emb.tp_size = 0, 1
emb._exl3_prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
emb.quant_method = m
with torch.device("cuda"):
    m.create_weights(emb, 160, [2048], 160, 2048, torch.bfloat16)
params = dict(emb.named_parameters())
for i in range(2):
    params[f"shard_{i}.trellis"].weight_loader(params[f"shard_{i}.trellis"], rows[i * 1024:(i + 1) * 1024])
for name, val in (("head_bias", head_bias), ("head_offsets", torch.arange(16, dtype=torch.int64) * 128),
                  ("head_vocab_sizes", torch.full((16,), 128, dtype=torch.int64)), ("layer_multipliers", torch.tensor([1, 2, 3]))):
    params[name].weight_loader(params[name], val)
m.process_weights_after_loading(emb)
print("ngram op name:", emb._exl3_opaque_name)
ids = torch.randint(0, 2048, (5, 16), device="cuda")
e_eager = m._embedding_impl(emb, ids)
e_op = m.embedding(emb, ids)
print("ngram eager vs op:", tuple(e_op.shape), e_op.dtype, "equal:", torch.equal(e_eager, e_op))
ok &= torch.equal(e_eager, e_op)


def f_emb(t):
    return m.embedding(emb, t)


e_c = torch.compile(f_emb, fullgraph=True)(ids)
print("ngram under torch.compile(fullgraph=True):", tuple(e_c.shape), "equal:", torch.equal(e_c, e_eager))
ok &= torch.equal(e_c, e_eager)
print("COMPILE_CHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
