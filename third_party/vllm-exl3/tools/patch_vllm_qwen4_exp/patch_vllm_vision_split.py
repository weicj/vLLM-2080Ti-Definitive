"""Drop the split vision q/k/v tensors an EXL3 pack ships next to the fused bf16 attn.qkv.

turboderp's Qwen3.8-Flash-Next pack stores `visual.blocks.N.attn.{q,k,v}_proj.*` trellis tensors
AND the original fused `attn.qkv.weight/bias` (bf16). vLLM's Qwen3_VisionTransformer holds only
`attn.qkv`, so AutoWeightsLoader raised "no module or parameter named 'blocks.0.attn.k_proj'"
(boot 74, after the whole language model had loaded). The fused bf16 copy is the better source
anyway. The substrings start with a dot, so `self_attn.q_proj` in the language model is untouched.
Backup <file>.orig2, idempotent, compile-checked.

usage: python3 patch_vllm_vision_split.py <site-packages/vllm>
"""
import os
import shutil
import sys

OLD = """            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
"""
NEW = """            orig_to_new_substr={
                "mtp.": None,
                # EXL3 packs ship split vision q/k/v trellis tensors next to
                # the fused bf16 attn.qkv this module holds; drop the split ones.
                ".attn.q_proj.": None,
                ".attn.k_proj.": None,
                ".attn.v_proj.": None,
            },
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
"""


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    path = os.path.join(
        sys.argv[1], "model_executor", "models", "qwen4_exp", "nvidia", "model.py"
    )
    if not os.path.isfile(path):
        print(f"ERROR: {path} not found")
        return 1
    src = open(path, encoding="utf-8").read()
    if '".attn.k_proj.": None' in src:
        print(f"already patched: {path}")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path}")
        return 1
    out = src.replace(OLD, NEW)
    compile(out, path, "exec")
    backup = path + ".orig2"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path} (backup {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
