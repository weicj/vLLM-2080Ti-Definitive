"""Correctness gate for Exl3EmbeddingMethod before any Qwen boot (GPU, no server).

1. Decoder: real packed rows from the pack (random shard blocks plus rows straddling
   every head boundary) decoded by exllamav3_ext.ngram_dequant, by the plugin's torch
   twin, and (if importable) by exllamav3's own ngram_codec.dequant_rows; all must agree.
2. Codebook: plugin's ngram_mul1_codebook vs exllamav3's mul1_codebook (if importable).
3. Method end to end on a fake 4 x 1024-row table filled with those real rows: shard
   params alias one table, loaders land by name, 2-D ids come back as [.., 160] in the
   layer dtype, ext and torch kernels agree, and the lookup captures into a CUDA graph.
4. vLLM's regenerated hash layout for ple_dense_layer_id 0 equals the checkpoint's.
5. The rewritten quantization_config resolves: n-gram spec for the ple prefix, K=5 lm_head.

Prints NGRAM_TEST: PASS only when every check passes.
usage: python3 test_ngram_embedding.py <pack_dir>
"""
import json
import os
import sys
import time

import torch

if len(sys.argv) < 2:
    sys.exit("usage: test_ngram_embedding.py <pack_dir>")
PACK = sys.argv[1]
PFX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
K = 5
ROWS_PER_SHARD = 2500012
NUM_SHARDS = 128
FAKE_SHARDS, FAKE_ROWS = 4, 1024
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


t0 = time.time()
dev = torch.device("cuda")
from safetensors import safe_open
import exllamav3_ext as ext
from vllm_exl3 import exl3 as X

print(f"plugin {X.__file__}")
check(hasattr(X, "Exl3EmbeddingMethod"), "plugin exposes Exl3EmbeddingMethod")
check(hasattr(ext, "ngram_dequant"), "exllamav3_ext has ngram_dequant")

# ---- 1. gather real rows -------------------------------------------------------
with safe_open(os.path.join(PACK, "ngram_embedding.safetensors"), "pt") as f:
    head_bias = f.get_tensor(PFX + "head_bias")
    head_offsets = f.get_tensor(PFX + "head_offsets")
    head_sizes = f.get_tensor(PFX + "head_vocab_sizes")
    layer_mult = f.get_tensor(PFX + "layer_multipliers")
    total = int(head_offsets[-1] + head_sizes[-1])
    g = torch.Generator().manual_seed(1234)
    ids = []
    blocks = []
    # rows straddling every head boundary (4 before, 4 after), then random 64-row blocks
    for h in range(1, 16):
        b = int(head_offsets[h])
        blocks.append((b - 4, 8))
    n_random = (FAKE_SHARDS * FAKE_ROWS - sum(n for _, n in blocks)) // 64
    for _ in range(n_random):
        start = int(torch.randint(0, total - 64, (1,), generator=g))
        if start % ROWS_PER_SHARD + 64 > ROWS_PER_SHARD:  # keep a block inside one shard
            start -= 64
        blocks.append((start, 64))
    rows = []
    for start, n in blocks:
        s, r = divmod(start, ROWS_PER_SHARD)
        assert r + n <= ROWS_PER_SHARD
        rows.append(f.get_slice(f"{PFX}shard_{s}.trellis")[r : r + n])
        ids.extend(range(start, start + n))
packed = torch.cat(rows, 0)
ids = torch.tensor(ids, dtype=torch.int64)
check(packed.shape[1] == 1 + 160 * K // 16 and packed.dtype == torch.int16,
      f"packed rows are int16 [{packed.shape[0]}, {packed.shape[1]}] (K={K})")
check(total == 320001446 and int(head_sizes.sum()) == total, f"checkpoint head layout sums to {total}")
pad = FAKE_SHARDS * FAKE_ROWS - packed.shape[0]
if pad:
    packed = torch.cat([packed, packed[:pad]], 0)
    ids = torch.cat([ids, ids[:pad]], 0)
heads = (torch.searchsorted(head_offsets, ids, right=True) - 1).clamp(0, 15)
print(f"  rows={packed.shape[0]} heads used={sorted(set(heads.tolist()))} read {time.time()-t0:.1f}s")

packed_d, heads_d, bias_d = packed.to(dev), heads.to(torch.int32).to(dev), head_bias.to(dev)
out_ext = torch.empty(packed.shape[0], 160, dtype=torch.float16, device=dev)
ext.ngram_dequant(packed_d.contiguous(), K, heads_d.contiguous(), bias_d.contiguous(), out_ext)
out_torch = X.ngram_dequant_rows_torch(packed_d, K, heads_d, bias_d)
diff = (out_ext.float() - out_torch.float()).abs()
check(torch.equal(out_ext, out_torch), f"ext == torch twin (max abs diff {diff.max().item():.3e}, mismatches {(diff > 0).sum().item()})")
check(out_ext.isfinite().all().item() and out_ext.abs().max().item() > 0, f"decoded rows finite and non-zero (|max| {out_ext.abs().max().item():.3f})")
try:
    from exllamav3.modules.quant.exl3_lib import ngram_codec as NC
    cb_ref = NC.mul1_codebook(dev)
    cb_mine = X.ngram_mul1_codebook(dev)
    check(torch.equal(cb_ref.to(torch.float16), cb_mine), f"codebook == exllamav3 mul1_codebook (mismatches {(cb_ref.to(torch.float16) != cb_mine).sum().item()})")
    ref = NC.dequant_rows(packed_d, K, cb_ref, bias_d.float()[heads_d.long()]).to(torch.float16)
    d2 = (ref.float() - out_ext.float()).abs()
    check(torch.equal(ref, out_ext), f"ext == exllamav3 dequant_rows (max abs diff {d2.max().item():.3e}, mismatches {(d2 > 0).sum().item()})")
except Exception as e:  # noqa: BLE001
    print(f"  skip exllamav3 ngram_codec reference: {type(e).__name__}: {str(e)[:100]}")

# ---- 3. method end to end on a fake table ----------------------------------------
cfg = X.Exl3Config(bits=3, codebook="mul1", scope="native_all_linears",
                   ngram_embedding={"bits": K, "num_shards": FAKE_SHARDS, "rows_per_shard": FAKE_ROWS,
                                    "num_heads": 16, "modules": ["ngram_embedding"]})
spec = cfg._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
check(spec is not None and cfg._ngram_embedding_spec("model.layers.1.ple.kv_proj") is None, "spec matches the table prefix only")
method = X.Exl3EmbeddingMethod(cfg, spec)
layer = torch.nn.Module()
layer.tp_rank, layer.tp_size = 0, 1
with torch.device(dev):
    method.create_weights(layer, 160, [FAKE_SHARDS * FAKE_ROWS], 160, FAKE_SHARDS * FAKE_ROWS, torch.bfloat16)
table = layer._exl3_ngram_table
check(table.device.type == "cuda" and table.dtype == torch.int16, f"table on {table.device} {table.dtype} {tuple(table.shape)}")
names = dict(layer.named_parameters())
check(all(f"shard_{i}.trellis" in names for i in range(FAKE_SHARDS)) and {"head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers"} <= set(names),
      f"named_parameters has shard_i.trellis and aux ({len(names)} params)")
check(all(names[f"shard_{i}.trellis"].data_ptr() == table[i].data_ptr() for i in range(FAKE_SHARDS)), "shard params alias the table (no second copy)")
for i in range(FAKE_SHARDS):
    p = names[f"shard_{i}.trellis"]
    p.weight_loader(p, packed[i * FAKE_ROWS : (i + 1) * FAKE_ROWS])
fake_offsets = torch.arange(16, dtype=torch.int64) * (FAKE_SHARDS * FAKE_ROWS // 16)
fake_sizes = torch.full((16,), FAKE_SHARDS * FAKE_ROWS // 16, dtype=torch.int64)
for name, val in (("head_bias", head_bias), ("head_offsets", fake_offsets), ("head_vocab_sizes", fake_sizes), ("layer_multipliers", layer_mult)):
    names[name].weight_loader(names[name], val)
bad = False
try:
    p = names["shard_0.trellis"]
    p.weight_loader(p, torch.zeros(FAKE_ROWS, 31, dtype=torch.int16))
except ValueError:
    bad = True
check(bad, "loader rejects a shard with the wrong K/words")
method.process_weights_after_loading(layer)
check(method.kernel == "ext" and method._ext is not None, f"kernel resolved: {method.kernel}")
check(torch.equal(layer._exl3_ngram_rows.cpu(), packed), "table content equals the loaded rows")
gen = torch.Generator(device="cpu").manual_seed(7)
q_ids = torch.randint(0, FAKE_SHARDS * FAKE_ROWS, (37, 16), generator=gen).to(dev)
out = method.embedding(layer, q_ids)
check(tuple(out.shape) == (37, 16, 160) and out.dtype == torch.bfloat16, f"embedding output {tuple(out.shape)} {out.dtype}")
flat = q_ids.reshape(-1)
ref_heads = (flat // (FAKE_SHARDS * FAKE_ROWS // 16)).to(torch.int32)
ref = X.ngram_dequant_rows_torch(packed_d[flat], K, ref_heads, bias_d).to(torch.bfloat16).view(37, 16, 160)
check(torch.equal(out, ref), "embedding() == torch reference on the same rows/heads")
method.kernel = "torch"
out2 = method.embedding(layer, q_ids)
method.kernel = "ext"
check(torch.equal(out, out2), "VLLM_EXL3_NGRAM_KERNEL=torch path == ext path")
# CUDA graph capture of the lookup (no host sync allowed on this path)
static_ids = q_ids.clone()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        method.embedding(layer, static_ids)
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out = method.embedding(layer, static_ids)
graph.replay()
torch.cuda.synchronize()
check(torch.equal(static_out, out), "lookup captured and replayed in a CUDA graph with identical output")
static_ids.copy_(torch.randint(0, FAKE_SHARDS * FAKE_ROWS, (37, 16), generator=gen).to(dev))
graph.replay()
torch.cuda.synchronize()
check(torch.equal(static_out, method.embedding(layer, static_ids)), "graph replay tracks new ids")

# ---- 4. vLLM hash layout for ple_dense_layer_id 0 ----------------------------------
try:
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding as NG
    sizes, offs, tot = NG._make_vocab_layout(ngram_vocab_size_base=20000000, ngram_heads=16, ple_dense_layer_id=0)
    mult = NG._make_layer_multipliers(ngram_size=3, unigram_vocab_size=248320, seed=1234, ple_dense_layer_id=0)
    check(sizes == head_sizes.tolist() and offs == head_offsets.tolist() and tot == total, "vLLM head layout (id 0) == checkpoint")
    check(mult == layer_mult.tolist(), "vLLM layer multipliers (id 0) == checkpoint")
except Exception as e:  # noqa: BLE001
    check(False, f"vLLM layout check raised {type(e).__name__}: {str(e)[:120]}")

# ---- 5. the rewritten config resolves ------------------------------------------------
qc = json.load(open(os.path.join(PACK, "config.json")))
qc = (qc.get("text_config") or {}).get("quantization_config") or qc.get("quantization_config")
c2 = X.Exl3Config.from_config(dict(qc))
sp = c2._ngram_embedding_spec("model.layers.1.ple.ple_embedding.ngram_embedding")
check(sp is not None and sp["bits"] == K and sp["num_shards"] == NUM_SHARDS and sp["rows_per_shard"] == ROWS_PER_SHARD and sp["num_heads"] == 16,
      f"config ngram spec: {sp}")
check(c2._matches_non_routed_exl3("lm_head") and c2._bits_for_non_routed("lm_head") == 5, "config lm_head -> K5 trellis linear")
check(not any(".shard_" in k for k in (c2.non_routed_exl3.get("layers") or {})), "no n-gram shard prefixes leaked into the dense map")

print(f"elapsed {time.time()-t0:.1f}s")
print("NGRAM_TEST: " + ("PASS" if not failures else f"FAIL ({len(failures)}): " + "; ".join(failures)[:600]))
