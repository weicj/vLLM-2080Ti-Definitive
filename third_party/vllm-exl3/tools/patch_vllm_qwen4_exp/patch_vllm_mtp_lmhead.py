"""quant_config on the Qwen4Exp MTP draft's ParallelLMHead.

The draft model shares the main checkpoint's lm_head (mapper: ``lm_head.`` names go to the
MTP too). Its ``ParallelLMHead`` was built without quant_config, so it expected ``lm_head.weight``
while the pack ships ``lm_head.{trellis,suh,svh,mul1}`` (boot 79). Same one-line change as the
main model. Backup <file>.orig, idempotent, compile-checked.

usage: python3 patch_vllm_mtp_lmhead.py <site-packages/vllm>
"""
import os
import shutil
import sys

OLD = """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
"""
NEW = """                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
"""


def main():
    path = os.path.join(sys.argv[1], "models", "qwen4_exp", "nvidia", "mtp.py")
    src = open(path, encoding="utf-8").read()
    if NEW in src:
        print(f"already patched: {path}")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"ERROR: anchor found {n} times in {path}")
        return 1
    out = src.replace(OLD, NEW)
    compile(out, path, "exec")
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    open(path, "w", encoding="utf-8").write(out)
    print(f"patched {path} (backup {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
