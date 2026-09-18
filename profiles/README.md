# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. They do not select the
checkpoint, GPUs, port, chat template, or reasoning defaults; those remain
launcher settings. Select the target with `MODEL_DIR` and the optional DFlash
draft with `SPECULATIVE_MODEL`.

Profiles are grouped by hardware first, then model family, weight format, and
startup mode:

```text
profiles/
  2x2080Ti/   # [hardware-specific guide](2x2080Ti/README.md)
  4xT10/      # [hardware-specific guide](4xT10/README.md)
  8xT10/      # [hardware-specific guide](8xT10/README.md)
```

Profile filenames use `<decoder>-<kv>-<concurrency><context>-<message>.env`.
For example, `dflash2-tqk8v4-2x172k-text-only.env` is a DFlash2 route using
TQK8V4 KV, two concurrent requests, 172K context per request, and text-only
messages. Decode routing uses `SPECULATIVE_METHOD=none|mtp|dflash` and
`SPECULATIVE_TOKENS`; the defaults are `0`, `3`, and `7`, respectively.
Per-request speculative metrics are a Launcher setting:
`PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`; DFlash defaults to
`detailed` until changed in Launcher.

Use `./launcher.sh --print-config` after selecting a profile to inspect the
resolved route before starting the service.

The eight-T10 Qwen3.8 Flash-Next PP profiles are documented in
[`8xT10/README.md`](8xT10/README.md). Their measured throughput is intentionally
left pending until the auditable warm-up plus three-sample runner writes a
manifest; historical numbers are not silently promoted into this catalog.
