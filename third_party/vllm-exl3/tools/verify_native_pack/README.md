# tools/verify_native_pack

Four GPU smoke tests against a real pack directory, meant to run before
starting a full vLLM server so a bad pack or a broken kernel change fails in
seconds instead of after a multi-minute boot. Each takes `<pack_dir>` as its
only argument and needs `vllm`, `vllm_exl3`, `exllamav3_ext`, and a GPU; none
of them start a server.

- `test_ngram_embedding.py` is the correctness gate for
  `Exl3EmbeddingMethod`. It decodes real packed rows (random shard blocks
  plus rows straddling every head boundary) three ways -- the
  `exllamav3_ext.ngram_dequant` kernel, the plugin's pure-torch fallback, and
  (if importable) exllamav3's own decoder -- and requires all three to agree;
  checks the plugin's mul1 codebook against exllamav3's; runs the embedding
  method end to end on a fake table (shard params aliasing, name-based
  loading, 2-D id lookups, ext/torch agreement, CUDA graph capture); and
  checks vLLM's regenerated hash layout and the rewritten
  `quantization_config` both resolve correctly. Prints `NGRAM_TEST: PASS`
  only if every check passes.

- `mul1_check.py` is pre-boot sanity for mul1-coded packs: it confirms the
  packed marker tensor equals the plugin's `MUL1_MARKER_SIGNED_INT32`
  constant, and that `LinearEXL3` actually forwards differently when built
  with a mul1 marker versus an mcg marker -- i.e. that the two codebooks are
  wired to distinct code paths, not silently interchangeable.

- `pad_check.py` is pre-boot sanity for padded EXL3 linears (matrices padded
  to a multiple of 128): it checks that a padded layer's extra output
  columns are exactly zero (`svh` is zero there) and that a padded layer
  accepts zero-extended input without changing the unpadded part of the
  result.

- `compile_check.py` checks that the EXL3 custom ops register with
  `torch.ops.vllm`, run eagerly, and trace under
  `torch.compile(fullgraph=True)` for both a padded dense linear and the
  n-gram lookup, since a custom op that only works eagerly breaks the
  moment vLLM compiles the model.

Usage: `python3 <script>.py <pack_dir>` for each.
