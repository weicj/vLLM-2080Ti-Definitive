"""Express a turboderp-native EXL3 pack in the vllm-exl3 plugin's own config vocabulary.

The plugin already understands a base K plus per-layer overrides (`layer_bits`) for experts, a
per-prefix bit map for dense linears (`non_routed_exl3.layers`), and a row-wise n-gram table
spec (`ngram_embedding`). A native pack with a fractional average (3.05 bpw) is just those maps
with a varying K, so no plugin code needs to change as long as every MoE layer's experts share
one K. This reads the header scan, checks that, and rewrites config.json's quantization_config
accordingly. The original block is preserved under `native_quantization_config` and
config.json is backed up to config.json.native first.

Refuses (exit 2) if any MoE layer has experts at different K, because the plugin stacks a
layer's experts into one tensor and cannot hold mixed widths, or if the pack holds n-gram
tables of differing geometry (the plugin takes one spec).

usage: python3 qwen_pack_config.py <pack_dir> [--scan pack_scan.json] [--dry-run]
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--scan", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    scan_path = args.scan or os.path.join(args.pack, "pack_scan.json")
    scan = json.load(open(scan_path))
    cfg_path = os.path.join(args.pack, "config.json")
    cfg = json.load(open(cfg_path))
    holder = cfg.get("text_config", cfg)
    q = holder.get("quantization_config") or cfg.get("quantization_config") or {}
    if q.get("native_quantization_config"):
        # config.json was already rewritten by this tool; start from the native block
        q = q["native_quantization_config"]

    if scan["expert_k_nonuniform"]:
        print("REFUSE: experts within a layer have different K; plugin cannot stack them:")
        for r in scan["expert_k_nonuniform"][:10]:
            print(f"  layer {r['layer']} {r['proj']}: {r['ks']}")
        return 2
    if scan.get("ngram_problems"):
        print("REFUSE: n-gram table problems:")
        for p in scan["ngram_problems"]:
            print("  ", p)
        return 2

    # one K per layer (gate/up/down must agree too, since the plugin uses one K per layer)
    layer_k = {}
    disagree = []
    for layer, row in scan["expert_k_per_layer"].items():
        ks = {row[p][0] for p in row}
        if len(ks) != 1:
            disagree.append((layer, row))
        else:
            layer_k[int(layer)] = ks.pop()
    if disagree:
        print("REFUSE: gate/up/down experts differ in K within a layer:")
        for layer, row in disagree[:10]:
            print(f"  layer {layer}: {row}")
        return 2

    base = collections.Counter(layer_k.values()).most_common(1)[0][0]
    layer_bits = {str(l): k for l, k in sorted(layer_k.items()) if k != base}

    # vLLM fuses some checkpoint projections into one module (packed_modules_mapping in
    # qwen4_exp), and its module prefixes differ from checkpoint names, so emit the map under
    # every plausible name: raw checkpoint prefix, the fused module it lands in, and both the
    # checkpoint root and the vLLM-side root. Extra keys are harmless; a missing one leaves a
    # layer unquantized and the load fails on a missing .weight.
    fused = {
        "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
        "in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz",
        "in_proj_b": "in_proj_ba", "in_proj_a": "in_proj_ba",
        "key_proj": "kv_proj", "value_proj": "kv_proj",
    }
    roots = [("model.language_model.", "language_model.model."),
             ("model.language_model.", "model."),
             ("model.visual.", "visual."), ("model.visual.", "vision_model.")]
    # vLLM numbers the MTP draft's layers after the main stack (checkpoint mtp.layers.0 is
    # module mtp.layers.<num_hidden_layers>), so emit those names too.
    n_main = int(holder.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    mtp_re = re.compile(r"^mtp\.layers\.(\d+)\.")
    dense_layers = {}
    suffixes = set()
    for prefix, k in scan["dense_k_by_prefix"].items():
        variants = {prefix}
        head, _, leaf = prefix.rpartition(".")
        if leaf in fused:
            variants.add(f"{head}.{fused[leaf]}")
        for v in list(variants):
            for src, dst in roots:
                if v.startswith(src):
                    variants.add(dst + v[len(src):])
        if n_main:
            for v in list(variants):
                m = mtp_re.match(v)
                if m:
                    variants.add(f"mtp.layers.{n_main + int(m.group(1))}." + v[m.end():])
        for v in variants:
            dense_layers[v] = {"bits": int(k)}
            tail = v.split(".")
            suffixes.add(".".join(tail[-2:]) if len(tail) >= 2 else v)

    codebook = str(q.get("codebook", "mcg"))
    k_counts = collections.Counter(int(k) for k in scan["dense_k_by_prefix"].values())
    dense_default = k_counts.most_common(1)[0][0]
    new_q = {
        "quant_method": "exl3",
        "bits": int(base),
        "codebook": codebook,
        "head_bits": int(q.get("head_bits", 16)),
        "scope": "native_all_linears",
        "layer_bits": layer_bits,
        "non_routed_exl3": {
            "codebook": codebook,
            "bits": int(dense_default),
            "modules": sorted(suffixes),
            "layers": dense_layers,
        },
        "native_quantization_config": q,
        "derived_from_headers": os.path.basename(scan_path),
    }

    # Row-wise n-gram tables: one spec, matched by module leaf name; all tables must agree.
    tables = scan.get("ngram_tables") or {}
    if tables:
        geoms = {(t["bits"], t["num_shards"], t["rows_per_shard"]) for t in tables.values()}
        if len(geoms) != 1:
            print("REFUSE: n-gram tables differ in geometry:", geoms)
            return 2
        heads = set()
        for t in tables.values():
            hb = t["aux"].get("head_bias")
            if not hb or len(hb["shape"]) != 2 or hb["shape"][1] != 160:
                print("REFUSE: n-gram table without a [heads, 160] head_bias:", t["aux"])
                return 2
            heads.add(int(hb["shape"][0]))
        if len(heads) != 1:
            print("REFUSE: n-gram tables differ in head count:", heads)
            return 2
        bits, num_shards, rows = next(iter(geoms))
        new_q["ngram_embedding"] = {
            "bits": int(bits),
            "num_shards": int(num_shards),
            "rows_per_shard": int(rows),
            "num_heads": heads.pop(),
            "modules": sorted({r.rsplit(".", 1)[-1] for r in tables}),
        }

    print(f"experts: base K={base}, layer overrides={len(layer_bits)} {layer_bits if len(layer_bits) < 20 else '(many)'}")
    print(f"dense linears mapped: {len(dense_layers)}; K distribution:",
          dict(collections.Counter(v['bits'] for v in dense_layers.values())))
    print(f"codebook={codebook} head_bits={new_q['head_bits']}")
    print("ngram_embedding:", new_q.get("ngram_embedding"))
    if args.dry_run:
        print("dry run, config.json untouched")
        return 0

    backup = cfg_path + ".native"
    if not os.path.exists(backup):
        shutil.copyfile(cfg_path, backup)
    if "text_config" in cfg and "quantization_config" in cfg["text_config"]:
        cfg["text_config"]["quantization_config"] = new_q
    cfg["quantization_config"] = new_q
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"written {cfg_path} (backup {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
