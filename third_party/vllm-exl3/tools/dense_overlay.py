#!/usr/bin/env python3
"""Build a dense-EXL3 overlay pack for GLM-5.3-Flash.

The source pack (routed experts already EXL3, everything else BF16) is left untouched:
the overlay directory symlinks every file of it and adds one safetensors file holding
EXL3 tensors for the non-expert linears, range-read from a full EXL3 quant on the Hub,
plus a rewritten index and a config.json carrying the ``non_routed_exl3`` block that
vllm-exl3 reads. No shard of the source pack is rewritten and nothing is quantized here.

    dense_overlay.py --branch 2.05bpw --src <pack> --out <overlay> --dry-run
    dense_overlay.py --branch 2.05bpw --src <pack> --out <overlay>
    dense_overlay.py --branch 2.05bpw --src <pack> --out <overlay> --verify
    # stack the MTP draft layer on an existing overlay pack (draft module prefixes are model.layers.N)
    dense_overlay.py --branch 2.05bpw --src <overlay> --out <overlay-mtp> --tag -mtp --skip-layers ""         --draft-layers 45 --draft-prefix-rewrite model.language_model.:model.
"""

import argparse
import collections
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

REPO = "https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/resolve/{branch}/"
MARKERS = {"mul1": -2082680531, "mcg": -877912083}  # signed int32 of 0x83DCD12D / 0xCBAC1FED
DTYPE_SIZE = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4}
CHUNK = 8 << 20

# pack tensor suffix -> (module suffix the serving model uses, shard index in that module)
FORK = {
    "self_attn.o_proj": ("self_attn.o_proj", None),
    "self_attn.q_b_proj": ("self_attn.q_b_proj", None),
    "mlp.down_proj": ("mlp.down_proj", None),
    "mlp.shared_experts.down_proj": ("mlp.shared_experts.down_proj", None),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
    "mlp.shared_experts.gate_proj": ("mlp.shared_experts.gate_up_proj", 0),
    "mlp.shared_experts.up_proj": ("mlp.shared_experts.gate_up_proj", 1),
    "self_attn.q_a_proj": ("self_attn.fused_qkv_a_proj", 0),
    "self_attn.kv_a_proj_with_mqa": ("self_attn.fused_qkv_a_proj", 1),
    "self_attn.q_proj": ("self_attn.in_proj_qkvbfg_a", 0),
    "self_attn.k_proj": ("self_attn.in_proj_qkvbfg_a", 1),
    "self_attn.v_proj": ("self_attn.in_proj_qkvbfg_a", 2),
}
# modules whose remaining shards stay BF16 in the serving model (b_proj, f_a_proj, g_a_proj)
BF16_SHARDS = {"self_attn.in_proj_qkvbfg_a": [3, 4, 5]}
# pack tensors that the Hub quant stores fused: suffix -> (fused suffix, block index, block count)
FUSED_SOURCE = {
    "self_attn.q_proj": ("self_attn.qkv_proj", 0, 3),
    "self_attn.k_proj": ("self_attn.qkv_proj", 1, 3),
    "self_attn.v_proj": ("self_attn.qkv_proj", 2, 3),
}


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- http
def http_get(url, rng=None, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dense_overlay/1"})
            if rng is not None:
                req.add_header("Range", "bytes=%d-%d" % (rng[0], rng[1]))
            resp = urllib.request.urlopen(req, timeout=60)
            if rng is not None and resp.status != 206:
                raise RuntimeError("expected 206 for a ranged read, got %d" % resp.status)
            return resp
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def http_bytes(url, rng=None):
    return http_get(url, rng).read()


def http_stream_to(url, rng, out, expected):
    """Stream one byte range into an open file; returns bytes written."""
    resp = http_get(url, rng)
    n = 0
    while True:
        buf = resp.read(CHUNK)
        if not buf:
            break
        out.write(buf)
        n += len(buf)
    if n != expected:
        raise RuntimeError("short read for %s: %d of %d bytes" % (url, n, expected))
    return n


# ---------------------------------------------------------------------- safetensors
def parse_header(blob):
    hlen = struct.unpack("<Q", blob[:8])[0]
    return hlen, json.loads(blob[8:8 + hlen].decode("utf-8"))


def local_headers(pack):
    idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))
    headers = {}
    for fn in sorted(set(idx["weight_map"].values())):
        with open(os.path.join(pack, fn), "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            headers[fn] = (hlen, json.loads(f.read(hlen).decode("utf-8")))
    return idx, headers


def remote_headers(base, cache_path):
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    idx = json.loads(http_bytes(base + "model.safetensors.index.json"))
    headers = {}
    for fn in sorted(set(idx["weight_map"].values())):
        hlen = struct.unpack("<Q", http_bytes(base + fn, (0, 7)))[0]
        headers[fn] = [hlen, json.loads(http_bytes(base + fn, (8, 8 + hlen - 1)).decode("utf-8"))]
        log("  header %s (%d tensors)" % (fn, len(headers[fn][1]) - 1))
    data = {"weight_map": idx["weight_map"], "headers": headers}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(data, open(cache_path, "w"))
    return data


def tensor_range(hlen, meta):
    """Absolute byte range [start, end] of a tensor inside its safetensors file."""
    a, b = meta["data_offsets"]
    return 8 + hlen + a, 8 + hlen + b - 1


def nbytes(meta):
    n = DTYPE_SIZE[meta["dtype"]]
    for d in meta["shape"]:
        n *= d
    return n


def write_header(out, entries, metadata):
    header = {"__metadata__": metadata}
    off = 0
    for name, dtype, shape in entries:
        n = DTYPE_SIZE[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    out.write(struct.pack("<Q", len(blob)))
    out.write(blob)
    return header


# ----------------------------------------------------------------------------- plan
def build_plan(args, local_idx, local_hdr, remote):
    """One entry per replaced source tensor, in output order."""
    rmap, rhdr = remote["weight_map"], remote["headers"]

    def rmeta(name):
        fn = rmap.get(name)
        if fn is None:
            return None, None, None
        hlen, hdr = rhdr[fn]
        return fn, hlen, hdr[name]

    layer_re = args.root + "layers."
    plan, fork_keys, problems = [], collections.OrderedDict(), []
    for name, fn in sorted(local_idx["weight_map"].items()):
        if not name.startswith(layer_re) or not name.endswith(".weight"):
            continue
        rest = name[len(layer_re):]
        layer, suffix = rest.split(".", 1)
        suffix = suffix[: -len(".weight")]
        if int(layer) in args.skip_layers or suffix not in FORK:
            continue
        out_f, in_f = local_hdr[fn][1][name]["shape"]
        base = name[: -len(".weight")]
        src_suffix, block, nblocks = FUSED_SOURCE.get(suffix, (suffix, 0, 1))
        src = layer_re + layer + "." + src_suffix
        marker = next((m for m in MARKERS if rmap.get(src + "." + m)), None)
        parts = {}
        for part in ("trellis", "suh", "svh", marker):
            if part is None:
                continue
            fn_r, hlen, meta = rmeta(src + "." + part)
            if meta is None:
                problems.append("missing remote tensor %s.%s" % (src, part))
                break
            parts[part] = (fn_r, hlen, meta)
        if len(parts) != 4:
            problems.append("remote %s lacks a codebook marker or a part" % src)
            continue
        tsh = parts["trellis"][2]["shape"]
        k = tsh[2] // 16
        want = [in_f // 16, out_f * nblocks // 16, 16 * k]
        if tsh != want or parts["suh"][2]["shape"] != [in_f] or parts["svh"][2]["shape"] != [out_f * nblocks]:
            problems.append("%s: remote shapes trellis=%s suh=%s svh=%s vs local [%d,%d]" % (
                src, tsh, parts["suh"][2]["shape"], parts["svh"][2]["shape"], out_f, in_f))
            continue
        fork_suffix, shard = FORK[suffix]
        key = layer_re + layer + "." + fork_suffix
        rewrite = args.draft_prefix_rewrite if int(layer) in args.draft_layers else args.prefix_rewrite
        if rewrite:
            old, new = rewrite
            if key.startswith(old):
                key = new + key[len(old):]
        entry = fork_keys.setdefault(key, {"bits": k})
        if entry["bits"] != k:
            problems.append("%s: fused shards disagree on K (%d vs %d)" % (key, entry["bits"], k))
        if fork_suffix in BF16_SHARDS:
            entry["bf16_shards"] = BF16_SHARDS[fork_suffix]
        plan.append({
            "name": name, "base": base, "src": src, "block": block, "nblocks": nblocks,
            "k": k, "in": in_f, "out": out_f, "marker": marker,
            "parts": {p: {"file": v[0], "hlen": v[1], "meta": v[2]} for p, v in parts.items()},
        })
    return plan, fork_keys, problems


def plan_outputs(plan):
    """Output tensors (name, dtype, shape) in file order, plus how to obtain each."""
    outs = []
    for e in plan:
        b, k = e["base"], e["k"]
        outs.append((b + ".trellis", "I16", [e["in"] // 16, e["out"] // 16, 16 * k], e, "trellis"))
        outs.append((b + ".suh", "F16", [e["in"]], e, "suh"))
        outs.append((b + ".svh", "F16", [e["out"]], e, "svh"))
        outs.append((b + "." + e["marker"], "I32", [], e, e["marker"]))
    return outs


def download_bytes(e, part, base_url, cache):
    """Return the bytes of one output part, slicing fused sources when needed."""
    p = e["parts"][part]
    url = base_url + p["file"]
    rng = tensor_range(p["hlen"], p["meta"])
    if e["nblocks"] == 1 or part in ("suh", e["marker"]):
        return http_bytes(url, rng)
    key = (e["src"], part)
    if key not in cache:
        cache.clear()
        cache[key] = http_bytes(url, rng)
    blob = cache[key]
    j, nb = e["block"], e["nblocks"]
    if part == "svh":
        w = e["out"] * 2
        return blob[j * w:(j + 1) * w]
    # trellis [in/16, nb*out/16, 16K] row-major int16: block j is a contiguous run per row
    rows, cols, words = p["meta"]["shape"]
    rowb = cols * words * 2
    bw = (cols // nb) * words * 2
    return b"".join(blob[r * rowb + j * bw: r * rowb + (j + 1) * bw] for r in range(rows))


# ------------------------------------------------------------------------------ io
def link_pack(src, out, overlay_name):
    os.makedirs(out, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if fn in ("config.json", "model.safetensors.index.json") or fn == overlay_name:
            continue
        dst = os.path.join(out, fn)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.abspath(os.path.join(src, fn)), dst)


def write_pack_meta(args, local_idx, plan, fork_keys, header, overlay_name, branch):
    replaced = {e["name"] for e in plan}
    wmap = {n: f for n, f in local_idx["weight_map"].items() if n not in replaced}
    for n in header:
        if n != "__metadata__":
            wmap[n] = overlay_name
    idx = dict(local_idx)
    idx["weight_map"] = dict(sorted(wmap.items()))
    json.dump(idx, open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    q = cfg.setdefault("quantization_config", {})
    # a source that is itself an overlay pack keeps its existing keys (stacked overlays)
    layers = dict(q.get("non_routed_exl3", {}).get("layers", {}))
    layers.update(fork_keys)
    q["non_routed_exl3"] = {
        "codebook": plan[0]["marker"],
        "source": "turboderp/GLM-5.3-Flash-exl3@" + branch,
        "layers": layers,
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)


def summarize(plan, fork_keys, problems):
    hist = collections.Counter()
    total = 0
    for e in plan:
        hist["%s K=%d" % (e["src"].split("layers.", 1)[1].split(".", 1)[1], e["k"])] += 1
        total += sum(nbytes(p["meta"]) for p in e["parts"].values()) // e["nblocks"]
    log("PLAN tensors=%d output_files=1 download_bytes=%.2f GB fork_keys=%d" % (len(plan), total / 1e9, len(fork_keys)))
    for k, v in sorted(hist.items()):
        log("  %-45s x%d" % (k, v))
    for k in list(fork_keys)[:5]:
        log("  key %s -> %s" % (k, fork_keys[k]))
    for p in problems:
        log("PROBLEM " + p)
    return total


def verify(args, plan, fork_keys, overlay_name, base_url):
    path = os.path.join(args.out, overlay_name)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
        ok, checked, bad = True, 0, []
        for name, dtype, shape, e, part in plan_outputs(plan):
            m = hdr.get(name)
            if m is None or m["dtype"] != dtype or m["shape"] != shape:
                bad.append("shape/dtype %s: %s" % (name, m))
                continue
            f.seek(8 + hlen + m["data_offsets"][0])
            data = f.read(m["data_offsets"][1] - m["data_offsets"][0])
            if part in MARKERS:
                val = struct.unpack("<i", data)[0]
                if val != MARKERS[part]:
                    bad.append("marker %s = %d" % (name, val))
            elif part in ("suh", "svh") or (part == "trellis" and checked < 3):
                # re-read the source bytes and compare (small vectors always, a few trellises)
                if data != download_bytes(e, part, base_url, {}):
                    bad.append("bytes differ from source: %s" % name)
                if part == "trellis":
                    checked += 1
    idx = json.load(open(os.path.join(args.out, "model.safetensors.index.json")))["weight_map"]
    for e in plan:
        if e["name"] in idx:
            bad.append("replaced weight still indexed: " + e["name"])
        for part in e["parts"]:
            if idx.get(e["base"] + "." + part) != overlay_name:
                bad.append("overlay tensor not indexed: %s.%s" % (e["base"], part))
    cfg = json.load(open(os.path.join(args.out, "config.json")))
    layers = cfg["quantization_config"].get("non_routed_exl3", {}).get("layers", {})
    if any(layers.get(k) != v for k, v in fork_keys.items()):
        bad.append("config layers block differs from plan (%d vs %d keys)" % (len(layers), len(fork_keys)))
    for b in bad[:20]:
        log("VERIFY_FAIL " + b)
    if bad:
        return False
    log("DENSE_OVERLAY_OK n_tensors=%d bytes=%d file=%s" % (len(hdr) - 1, os.path.getsize(path), path))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--src", help="source pack directory (experts EXL3, dense BF16)")
    ap.add_argument("--out", help="overlay pack directory to create")
    ap.add_argument("--root", default="model.language_model.", help="tensor-name root of the language model")
    ap.add_argument("--skip-layers", default="45", help="comma list of layer indices to leave untouched (MTP)")
    ap.add_argument("--prefix-rewrite", default=None, help="OLD:NEW rewrite of config key prefixes")
    ap.add_argument("--draft-layers", default="", help="comma list of layers served by the MTP draft module")
    ap.add_argument("--draft-prefix-rewrite", default=None, help="OLD:NEW rewrite of config key prefixes for --draft-layers")
    ap.add_argument("--tag", default="", help="suffix for the overlay file name (stack a second overlay on an overlay pack)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/dense_overlay"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--probe-remote-only", action="store_true")
    args = ap.parse_args()
    args.skip_layers = {int(x) for x in args.skip_layers.split(",") if x}
    args.prefix_rewrite = tuple(args.prefix_rewrite.split(":", 1)) if args.prefix_rewrite else None
    args.draft_layers = {int(x) for x in args.draft_layers.split(",") if x}
    args.draft_prefix_rewrite = tuple(args.draft_prefix_rewrite.split(":", 1)) if args.draft_prefix_rewrite else None
    base_url = REPO.format(branch=args.branch)
    overlay_name = "dense-exl3-%s%s.safetensors" % (args.branch, args.tag)

    remote = remote_headers(base_url, os.path.join(args.cache, "%s-headers.json" % args.branch))
    if args.probe_remote_only:
        hist = collections.Counter()
        for fn, (hlen, hdr) in remote["headers"].items():
            for n, m in hdr.items():
                if n.endswith(".trellis") and ".experts." not in n:
                    hist[m["shape"][2] // 16] += 1
        log("PROBE branch=%s non-expert trellis K histogram=%s" % (args.branch, dict(hist)))
        return 0
    if not (args.src and args.out):
        ap.error("--src and --out are required")

    local_idx, local_hdr = local_headers(args.src)
    plan, fork_keys, problems = build_plan(args, local_idx, local_hdr, remote)
    summarize(plan, fork_keys, problems)
    if problems:
        log("ABORT: %d problems" % len(problems))
        return 1
    if not plan:
        log("ABORT: nothing to replace")
        return 1
    if args.verify:
        return 0 if verify(args, plan, fork_keys, overlay_name, base_url) else 1
    if args.dry_run:
        log("DRY_RUN_OK")
        return 0

    link_pack(args.src, args.out, overlay_name)
    outs = plan_outputs(plan)
    path = os.path.join(args.out, overlay_name)
    tmp = path + ".partial"
    cache = {}
    t0 = time.time()
    with open(tmp, "wb") as f:
        header = write_header(f, [(n, d, s) for n, d, s, _, _ in outs],
                              {"format": "pt", "source": "turboderp/GLM-5.3-Flash-exl3@" + args.branch})
        done = 0
        for i, (name, dtype, shape, e, part) in enumerate(outs):
            expected = header[name]["data_offsets"][1] - header[name]["data_offsets"][0]
            p = e["parts"][part]
            if e["nblocks"] == 1 and part == "trellis":
                n = http_stream_to(base_url + p["file"], tensor_range(p["hlen"], p["meta"]), f, expected)
            else:
                data = download_bytes(e, part, base_url, cache)
                if len(data) != expected:
                    raise RuntimeError("%s: got %d bytes, expected %d" % (name, len(data), expected))
                f.write(data)
                n = len(data)
            done += n
            if part == "trellis" and i % 40 == 0:
                log("  %5d/%d %s %.2f GB %.0fs" % (i, len(outs), name, done / 1e9, time.time() - t0))
    os.replace(tmp, path)
    write_pack_meta(args, local_idx, plan, fork_keys, header, overlay_name, args.branch)
    log("WROTE %s (%.2f GB in %.0fs)" % (path, os.path.getsize(path) / 1e9, time.time() - t0))
    return 0 if verify(args, plan, fork_keys, overlay_name, base_url) else 1


if __name__ == "__main__":
    sys.exit(main())
