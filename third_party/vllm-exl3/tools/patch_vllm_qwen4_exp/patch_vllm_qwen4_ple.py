"""Two plumbing lines in vLLM's qwen4_exp so the quant config is consulted for
lm_head and for the PLE n-gram table.

  nvidia/model.py   ParallelLMHead(...)              gains quant_config=self.quant_config
  nvidia/ple_layer.py PLEVocabParallelEmbedding(...) gains quant_config=quant_config

Without these vLLM builds both layers unquantized regardless of --quantization,
so an EXL3 pack whose lm_head / n-gram rows are trellis tensors cannot load.
Backs up each file to <file>.orig (kept if present), idempotent, compile-checked.

usage: python3 patch_vllm_qwen4_ple.py <site-packages/vllm>
"""
import os
import shutil
import sys

MODEL_OLD = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
MODEL_NEW = '''        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
'''
PLE_OLD = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
'''
PLE_NEW = '''        self.ngram_embedding = PLEVocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            quant_config=quant_config,
            prefix=f"{prefix}.ngram_embedding",
'''


def patch(path: str, old: str, new: str) -> bool:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return False
    src = open(path, encoding="utf-8").read()
    if new in src:
        print(f"already patched: {path}")
        return True
    n = src.count(old)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path} (need 1)")
        return False
    out = src.replace(old, new)
    try:
        compile(out, path, "exec")
    except SyntaxError as e:
        print(f"ERROR: patched source does not compile: {e}")
        return False
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path} (backup {backup})")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = os.path.join(sys.argv[1], "models", "qwen4_exp", "nvidia")
    ok = patch(os.path.join(root, "model.py"), MODEL_OLD, MODEL_NEW)
    ok = patch(os.path.join(root, "ple_layer.py"), PLE_OLD, PLE_NEW) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
