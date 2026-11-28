# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Inkling profiles are stored as `profiles/<hardware>/inkling-2-mini/<route>.env`.
Profiles for models with multiple weight variants use `profiles/<hardware>/<model>/<weights>/<route>.env`.
Select a hardware guide for model weights, Profile files, and benchmarks.

- [2xT10](2xT10/README.md)
- [4xT10](4xT10/README.md)
- [2x2080Ti](2x2080Ti/README.md)

`./launcher.sh` selects the model, GPU topology, port, chat template, reasoning,
and mode. The default mode is `fast`. Use `MODEL_DIR` for target weights and
`SPECULATIVE_MODEL` for a DFlash draft. Preview the resolved route with
`./launcher.sh --print-config`. [Non-interactive launch](../docs/non-interactive-launch.md)

Set `PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed` in the launcher to
control per-request speculative decoding metrics.

## Profile format

The filename follows `<decoder>-<kv>-<concurrency>x<context>-<message>.env`.
Match it to the route values. The context label uses decimal thousands, so
`MAX_MODEL_LEN=262144` appears as `262K`. Set `SPECULATIVE_METHOD` to
`none`, `mtp`, `dflash`, or `dspark` and set `SPECULATIVE_TOKENS` to the
draft length. DSpark loads its draft from the official Inkling checkpoint.

Use one `KEY=value` per line. Required fields are `MODEL_FAMILY`,
`MODEL_VARIANT`, `QUANTIZATION`, `KV_CACHE_DTYPE`, `MAX_MODEL_LEN`,
`GPU_UTIL`, `MAX_NUM_SEQS`, `MESSAGE_TYPE`, `SPECULATIVE_METHOD`, and
`SPECULATIVE_TOKENS`. [Example Profile](2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env)

Validate local Profiles with `bash tools/validate_profiles.sh`.
The validator also lists supported optional fields. [Validator](../tools/validate_profiles.sh)
