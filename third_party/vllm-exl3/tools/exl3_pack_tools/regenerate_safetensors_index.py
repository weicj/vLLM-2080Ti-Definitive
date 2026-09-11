"""Rebuild model.safetensors.index.json from the safetensors files actually present.

A native EXL3 pack's tensors (trellis / suh / svh / mcg / mul1 / n-gram shards) are pulled from
turboderp's own shard files, and the resulting file set frequently no longer matches the index
that shipped with the source checkpoint: files get renamed, split, or dropped, and the index's
weight_map still names the old files. vLLM's loader trusts the index over `glob`, so a stale
index makes it skip or mis-source tensors silently rather than fail loudly.

This reads every `*.safetensors` header (the 8-byte little-endian length prefix, then that many
bytes of JSON -- no tensor data is touched) in a pack directory and writes a fresh
model.safetensors.index.json: `metadata.total_size` is the sum of each tensor's
`data_offsets` span, and `weight_map` names every tensor's file. The original index, if any, is
backed up to `model.safetensors.index.json.native` (not overwritten if that backup already
exists, so re-running is safe). Refuses (exit 2) if the same tensor name appears in more than
one shard, since the index format cannot represent that and something upstream is wrong.

usage: python3 regenerate_safetensors_index.py <pack_dir> [--dry-run]
"""
import argparse
import glob
import json
import os
import shutil
import struct
import sys


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.pack, "*.safetensors")))
    if not shards:
        print(f"ERROR: no *.safetensors files under {args.pack}")
        return 2

    weight_map = {}
    total_size = 0
    dupes = []
    for shard in shards:
        base = os.path.basename(shard)
        header = read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in weight_map:
                dupes.append((name, weight_map[name], base))
                continue
            weight_map[name] = base
            offsets = meta.get("data_offsets") or [0, 0]
            total_size += int(offsets[1]) - int(offsets[0])

    if dupes:
        print(f"REFUSE: {len(dupes)} tensor name(s) appear in more than one shard:")
        for name, first, second in dupes[:10]:
            print(f"  {name}: {first} and {second}")
        return 2

    print(f"scanned {len(shards)} shard(s), {len(weight_map)} tensor(s), total_size={total_size}")
    if args.dry_run:
        return 0

    index_path = os.path.join(args.pack, "model.safetensors.index.json")
    backup_path = index_path + ".native"
    if os.path.exists(index_path) and not os.path.exists(backup_path):
        shutil.copyfile(index_path, backup_path)
        print(f"backed up existing index to {backup_path}")

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
