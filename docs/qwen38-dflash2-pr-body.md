# DFlash2: Qwen3.8 NVFP4 SM75 profile

## Summary

Promote the first validated DFlash2 route for dual RTX 2080 Ti SM75:

- target: `nvidia/Qwen3.8-27B-NVFP4`
- draft: `incoai/Qwen3.8-27B-DFlash2`
- KV: TurboQuant K8V4
- speculative K: 7
- target graph: PIECEWISE, capture size 8
- draft attention: `TRITON_ATTN`
- context: 262144 tokens, text-only

The SM75 TurboQuant target path now preserves causal verification sequence
lengths for every proposed token. B=8 is handled as safe causal chunks, which
keeps the graph route usable without changing verification semantics.

## Profile

```text
profiles/qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env
```

The profile does not contain host-specific GPU, port, chat-template, or
reasoning settings. Those remain launcher/service settings.

## Validation

Fixed high-acceptance 4K/128 measurements:

```text
155.164865 tok/s
166.566925 tok/s
166.756140 tok/s
mean: 162.829310 tok/s
```

Additional evidence:

- 3/3 correctness smokes returned `PROFILE_OK`.
- 3/3 speed streams returned 128/128 tokens and HTTP 200.
- A 262016-token prompt completed 16/16 output tokens.
- Real 4K and 8K output-stage HTML/JavaScript generations completed without
  stream interruption; the 8K result reached the requested length limit and
  contained Canvas/JavaScript code.
- The service remained alive with no new CUDA illegal instruction, EngineDead,
  fatal error, or stream failure.
- Target-runtime DFlash2 unit suite: `17 passed in 10.84s`.

Full validation details are in
[`qwen38-dflash2-profile-validation.md`](qwen38-dflash2-profile-validation.md).

## Verification commands

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile vllm/envs.py \
  vllm/v1/attention/backends/turboquant_attn.py
git diff --check
```

On the target runtime:

```bash
python -m pytest -q tests/v1/spec_decode/test_dflash2.py
```

## Scope notes

The repository worktree contains unrelated user changes in other runtime and
model routes. This PR body identifies the DFlash2/profile slice; those changes
must remain separate when creating a final git commit or external PR.
