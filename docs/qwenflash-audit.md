# Qwen Flash-Next audit protocol

`tools/benchmark_qwenflash_audit.py` measures a running Qwen3.8 Flash-Next
NVFP4 service without taking ownership of its process. For each shape it runs
one warm-up followed by three measured requests:

| Shape | Prompt | Completion | Samples |
|---|---:|---:|---:|
| `4k128` | 4096 | 128 | 1 warm-up + 3 measured |
| `32k512` | 32768 | 512 | 1 warm-up + 3 measured |

The helper uses the completions endpoint with a pure filler prompt and a
single allowed token. It therefore checks exact prompt/completion token counts,
HTTP 200, completed stream, and positive prefill/decode rates. A failed check
does not produce a valid audit result.

Use `--python /path/to/runtime/.venv/bin/python` when the benchmark dependencies
are installed in a runtime virtualenv; the default is the interpreter running
the audit script.

The JSON manifest records the UTC interval, relative profile and all route
keys, model path, server URL, git commit/branch/dirty state, GPU inventory,
benchmark contract, the effective KV cache dtype, and every raw
`profile_request.py` record. Profiles may use `float16` or `fp8` KV, but the
dtype is part of the result identity and must not be compared across runs
without calling out the difference. The profile is also required to declare
`MTP_K=0`, `SPECULATIVE_METHOD=none`, and `PLE_PLACEMENT=disk`; this prevents
accidentally mixing MTP, EXL3, or an unrecorded PLE placement into the
comparison.

The runner does not start or stop vLLM. Start each profile with the intended
GPU selection and port, run the helper, then stop the service using the normal
launcher lifecycle. Keep manifests outside the source tree (for example under
`results/qwenflash/`) so model outputs and logs are not committed.

Historical measurements may be cited as context, but only a manifest produced
by this protocol is current promotion evidence. Report median and mean for the
three measured samples, plus the GPU summary captured during each request.

## Historical baseline (pre-audit protocol)

The repository contains an earlier `.31` validation record from the
`feat/qwen38-flash-next-nvfp4` / PP-profile work (commits `6fdb2769f6` and
`052ce5223a`). It is retained here so a failed run on a newer runtime is not
mistaken for a historical lack of support:

| Route | Runtime context | GPU KV tokens | 4K/128 prefill / decode |
|---|---:|---:|---:|
| TP4xPP2, conservative util | 90K / 0.92 | 121,139 | 1,615.30 / 21.99 tok/s |
| TP4xPP2, high-capacity run | 90K / 0.96 | 100,031 | 1,697.99 / 28.18 tok/s |
| TP2xPP4 | 100K / 0.90 | 131,872 | 1,979.79 / 10.74 tok/s |

The corresponding `.31` artifacts are
`/home/max/workspace/VLLM-2080ti-0.2.1-pre/run-logs/`
`vllm-qwen38-flash-next-nvfp4-tp4pp2-20260909-004303.log` and
`vllm-qwen38-flash-next-nvfp4-tp2pp4-20260909-003753.log`; a second TP4xPP2
startup record is in
`/home/max/workspace/VLLM-2080ti-qwen38-flash-pr/run-logs/`
`vllm-qwen38-flash-next-nvfp4-tp4pp2-20260908-123049.log`.
They show successful TP4xPP2 startup at 90K and 100K maximum context. These numbers predate the current fixed-token manifest
contract and are not promotion evidence for the current branch. No matching,
auditable 32K/512 result or 35 tok/s TP4xPP2 result was found in the checked-in
records; those remain explicit follow-up targets.
