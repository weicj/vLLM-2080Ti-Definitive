"""Inventory a turboderp-native EXL3 pack's tensors so the plugin's config can be derived.

The vllm-exl3 plugin stacks a layer's experts into one [E, in/16, out/16, 16*K] tensor, so it
needs a single K per MoE layer. exl3 packs quantized to a fractional average (3.05 bpw) assign K
per tensor. This reads every safetensors header (no tensor data) and reports:
  - per MoE layer: the set of K values across experts for gate/up/down (uniform or not)
  - per dense linear (attention, dense MLP, shared expert, lm_head): K and suffixes
  - row-wise n-gram embedding tables (exllamav3 ngram format): shard count, rows, K, aux tensors
  - every tensor suffix seen per module family, so extra tensors the plugin ignores show up

K for a linear trellis tensor is shape[-1] // 16. An n-gram table row is 1 + 160*K/16 int16
words, so for `<root>.shard_N.trellis` [rows, words] K = (words - 1) * 16 // 160; those
tensors are reported under `ngram_tables` and kept out of the dense-linear map.
Writes a JSON summary next to stdout.

usage: python3 qwen_pack_scan.py <pack_dir> [--out scan.json]
"""
import argparse
import collections
import glob
import json
import os
import re
import struct
import sys

NGRAM_ROW_DIM = 160


def headers(pack):
    for shard in sorted(glob.glob(os.path.join(pack, "*.safetensors"))):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            yield os.path.basename(shard), name, meta.get("dtype"), meta.get("shape") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.pack, "pack_scan.json")

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    expert_re = re.compile(r"\.experts\.(\d+)\.(\w+)\.(\w+)$")
    ngram_re = re.compile(r"^(.*)\.shard_(\d+)\.trellis$")
    tensors = list(headers(args.pack))

    # n-gram tables first: roots are discovered from shard trellis names, then every
    # tensor directly under a root that is not a shard is an aux tensor of that table.
    ngram = {}
    for _, name, dtype, shape in tensors:
        m = ngram_re.match(name)
        if m and len(shape) == 2:
            t = ngram.setdefault(m.group(1), {"shards": set(), "shapes": set(), "dtypes": set(), "aux": {}})
            t["shards"].add(int(m.group(2)))
            t["shapes"].add((int(shape[0]), int(shape[1])))
            t["dtypes"].add(dtype)
    for _, name, dtype, shape in tensors:
        for root, t in ngram.items():
            if name.startswith(root + ".") and ".shard_" not in name[len(root):]:
                t["aux"][name[len(root) + 1:]] = {"dtype": dtype, "shape": [int(s) for s in shape]}
    ngram_tables = {}
    ngram_problems = []
    for root, t in ngram.items():
        shards = sorted(t["shards"])
        if shards != list(range(len(shards))):
            ngram_problems.append(f"{root}: shard indices not contiguous ({shards[:5]}...)")
        if len(t["shapes"]) != 1 or t["dtypes"] != {"I16"}:
            ngram_problems.append(f"{root}: shard shapes/dtypes not uniform: {t['shapes']} {t['dtypes']}")
            continue
        rows, words = next(iter(t["shapes"]))
        if (words - 1) * 16 % NGRAM_ROW_DIM:
            ngram_problems.append(f"{root}: words={words} is not 1 + 160*K/16")
            continue
        ngram_tables[root] = {
            "num_shards": len(shards),
            "rows_per_shard": rows,
            "words": words,
            "bits": (words - 1) * 16 // NGRAM_ROW_DIM,
            "total_rows": len(shards) * rows,
            "aux": t["aux"],
        }
    ngram_roots = tuple(ngram)

    families = collections.defaultdict(set)      # module family -> suffixes seen
    expert_k = collections.defaultdict(lambda: collections.defaultdict(set))  # layer -> proj -> {K}
    dense_k = {}                                   # module prefix -> K
    misc = collections.Counter()
    total = 0
    for shard, name, dtype, shape in tensors:
        total += 1
        if ngram_roots and name.startswith(tuple(r + "." for r in ngram_roots)):
            fam = re.sub(r"\.\d+\.", ".N.", name.rsplit(".", 1)[0])
            fam = re.sub(r"\.shard_\d+$", ".shard_N", fam)
            families[fam].add(name.rsplit(".", 1)[-1])
            continue
        if name.endswith(".trellis"):
            k = int(shape[-1]) // 16 if shape else None
        else:
            k = None
        m = expert_re.search(name)
        lm = layer_re.search(name)
        if m and lm:
            layer, proj, suffix = int(lm.group(1)), m.group(2), m.group(3)
            families[f"layers.N.experts.E.{proj}"].add(suffix)
            if k is not None:
                expert_k[layer][proj].add(k)
            continue
        base = name.rsplit(".", 1)[0]
        suffix = name.rsplit(".", 1)[-1]
        fam = re.sub(r"\.\d+\.", ".N.", base)
        families[fam].add(suffix)
        if k is not None:
            dense_k[base] = k
        if not lm and not name.startswith("mtp"):
            misc[fam] += 1

    per_layer = {}
    nonuniform = []
    for layer in sorted(expert_k):
        row = {}
        for proj in sorted(expert_k[layer]):
            ks = sorted(expert_k[layer][proj])
            row[proj] = ks
            if len(ks) != 1:
                nonuniform.append((layer, proj, ks))
        per_layer[layer] = row

    dense_by_fam = collections.defaultdict(collections.Counter)
    for base, k in dense_k.items():
        dense_by_fam[re.sub(r"\.\d+\.", ".N.", base)][k] += 1

    summary = {
        "pack": args.pack,
        "tensors": total,
        "moe_layers": len(per_layer),
        "expert_k_per_layer": {str(l): r for l, r in per_layer.items()},
        "expert_k_nonuniform": [{"layer": l, "proj": p, "ks": ks} for l, p, ks in nonuniform],
        "dense_k_by_family": {f: dict(c) for f, c in sorted(dense_by_fam.items())},
        "dense_k_by_prefix": dense_k,
        "ngram_tables": ngram_tables,
        "ngram_problems": ngram_problems,
        "suffixes_by_family": {f: sorted(s) for f, s in sorted(families.items())},
        "non_layer_families": dict(misc),
    }
    json.dump(summary, open(out, "w"), indent=1)

    print(f"tensors={total} moe_layers={len(per_layer)} nonuniform_expert_layers={len(nonuniform)}")
    kd = collections.Counter()
    for l, r in per_layer.items():
        for p, ks in r.items():
            kd[tuple(ks)] += 1
    print("expert K distribution (per layer/proj):", dict(kd))
    if nonuniform:
        print("NONUNIFORM expert K within a layer (plugin cannot stack these):")
        for l, p, ks in nonuniform[:12]:
            print(f"  layer {l} {p}: {ks}")
    print("dense K by family:")
    for f, c in sorted(dense_by_fam.items()):
        print(f"  {f}: {dict(c)}")
    print("n-gram tables:")
    for root, t in ngram_tables.items():
        print(f"  {root}: {t['num_shards']} shards x {t['rows_per_shard']} rows, words={t['words']} K={t['bits']}, aux={sorted(t['aux'])}")
    for p in ngram_problems:
        print("  PROBLEM:", p)
    print("suffixes by family:")
    for f, s in sorted(families.items()):
        print(f"  {f}: {sorted(s)}")
    print("written:", out)


if __name__ == "__main__":
    sys.exit(main())
