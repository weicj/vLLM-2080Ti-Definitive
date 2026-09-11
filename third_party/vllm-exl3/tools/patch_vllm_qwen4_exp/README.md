# tools/patch_vllm_qwen4_exp

Three small, idempotent patches to vLLM's own `Qwen4ExpForConditionalGeneration`
model code (`vllm/model_executor/models/qwen4_exp/nvidia/`), needed so the
plugin's `get_quant_method` is actually consulted for the layers a native EXL3
pack quantizes outside the routed experts. Each script takes the path to a
vLLM `site-packages/vllm` directory, checks whether it has already been
applied, compile-checks the patched source before writing it, and keeps a
backup of the original file (`.orig` / `.orig2`) so the patch can be reverted
by hand.

- `patch_vllm_qwen4_ple.py` adds `quant_config=` to the `ParallelLMHead` and
  `PLEVocabParallelEmbedding` constructors in `model.py` and `ple_layer.py`.
  Without it vLLM builds both layers unquantized regardless of
  `--quantization`, so a pack whose `lm_head` and n-gram table rows are EXL3
  trellis tensors cannot load: the quant config never gets asked for a method,
  and the layer expects a plain `lm_head.weight` / embedding tensor that the
  checkpoint does not have.

- `patch_vllm_vision_split.py` drops the split vision `q_proj` / `k_proj` /
  `v_proj` name substrings from `Qwen3_VisionTransformer`'s checkpoint
  name-mapping. turboderp's Qwen3.8-Flash-Next pack ships both the split
  trellis tensors and the original fused bf16 `attn.qkv.weight/bias`, but
  vLLM's vision transformer only holds the fused tensor, so
  `AutoWeightsLoader` raised `no module or parameter named
  'blocks.0.attn.k_proj'` once the language model had already finished
  loading. Dropping the split names is correct either way: the fused bf16
  copy is the one vLLM's module actually uses.

- `patch_vllm_mtp_lmhead.py` is the same `quant_config=` addition as the PLE
  patch, applied to the MTP draft model's `ParallelLMHead` in `mtp.py`. The
  draft shares the main checkpoint's `lm_head` weights (the checkpoint name
  mapper routes `lm_head.` to both), so it needs the same fix or the draft's
  head fails to load with the same missing-`lm_head.weight` error.

Usage: `python3 patch_vllm_qwen4_ple.py <site-packages/vllm>`, then the other
two the same way. Safe to run more than once.
