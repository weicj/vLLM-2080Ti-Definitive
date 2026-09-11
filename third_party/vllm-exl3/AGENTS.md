# Agent notes for vllm-exl3

Guidance for coding agents (and humans) working in this repo.

## What this is
An out-of-tree vLLM plugin registering `--quantization exl3` for EXL3
(ExLlamaV3 trellis, MCG codebook) packs. Routed MoE experts stay packed and run
through `exllamav3_ext` kernels. Canonical home of the plugin; the HF
`spark-vllm` repos mirror built wheels for fast installs.

## Ground rules
- Commits are authored by the repo owner only. Never add AI/tool attribution,
  Co-Authored-By trailers, or session links to commits, code, or docs.
- Never commit tokens or credentials. Grep your diff before committing.
- Do not document how EXL3 packs are *produced* — this repo is serving-side
  only. Quantization pipelines are out of scope on purpose.
- Keep `NOTICE`, `LICENSE`, `CITATION.cff`, and the README attribution intact.

## Layout
- `src/vllm_exl3/exl3.py` — the whole plugin: `Exl3Config` (quant registration,
  `layer_bits` per-layer bitrates, `non_routed_quantization` delegation,
  `non_routed_exl3` dense-linear matching), `Exl3MoEMethod` (packed
  create/load/apply), `Exl3LinearMethod` (dense EXL3 linears: per-shard
  `LinearEXL3`, mixed EXL3/BF16 shards, stale `.weight` discard), fused-kernel dispatch with the
  fat-expert cap `TEMP_ROWS_FUSED` (2048 — do not lower it; 128 caused a
  >163k-token prefill stall via the per-expert fallback loop).
- `src/glm53_exl3_plugin/` — deprecated compat shim. Must keep working for both
  `import glm53_exl3_plugin` and `from glm53_exl3_plugin.exl3 import ...`
  (submodule imports bypass lazy `__getattr__` — that was the 0.2.1 fix).

## Config contract (what packs must declare)
`quantization_config`: `quant_method: "exl3"`, `bits`, `codebook: "mcg"`,
optional `layer_bits` (per-layer overrides for mixed-bitrate packs), optional
`non_routed_quantization` (dict of the source-format quant config for
non-routed weights, e.g. DeepSeek block-FP8), optional `non_routed_exl3`
(`layers` map keyed by vLLM module prefix with `bits` and `bf16_shards`, or
the `modules`/`bits`/`layer_bits` suffix form; see README). `tools/dense_overlay.py`
builds an overlay pack directory (symlinks + one extra safetensors file +
rewritten index/config) from tensors that already exist in a full EXL3 quant;
it moves tensors, it does not quantize. A pack whose config declares a
different quant_method gets silently overridden by vLLM — always fix the pack
config, never work around it in code.

## Build / test
- Native build (DGX Spark GB10 / Blackwell): `pip install -e . --no-build-isolation` or `python setup.py build_ext --inplace` compiles `vllm_exl3_c` native kernels.
- Pure Python wheel build (optional fallback): `VLLM_EXL3_NO_CUDA=1 python -m build --wheel` produces `py3-none-any` wheel for environments without CUDA compilers.
- Unit tests: `python -m pytest` executes tests, auto-skipping CUDA kernels when running in CPU or non-compiled environments.
- Dense-linear tests (need a GPU + exllamav3, no pytest):
  `python tests/test_exl3_linear.py` prints `EXL3_LINEAR_TP_TEST PASS`,
  `EXL3_LINEAR_TEST PASS`, `EXL3_LINEAR_MIXED_TEST PASS`.
- Import gates after any packaging change:
  `python -c "import vllm_exl3; from glm53_exl3_plugin.exl3 import Exl3Config"`.
- Real serving tests need a GB10 host with the fork runtime + exllamav3;
  see the recipe repos for the serve scripts and the memory-census diagnostic.

## Versioning
Entry point is `vllm_exl3 = vllm_exl3:register` — exactly one; a second entry
point double-registers the quant method. Tag releases `vX.Y.Z`, attach the
wheel, and mirror it to the HF spark-vllm repo **replacing** the previous wheel
(two versions in that repo break `pip install dir/*.whl`).
