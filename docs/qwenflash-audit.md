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
benchmark contract, and every raw `profile_request.py` record. The profile is
required to declare FP16 KV, `MTP_K=0`, `SPECULATIVE_METHOD=none`, and
`PLE_PLACEMENT=disk`; this prevents accidentally mixing MTP, EXL3, or an
unrecorded PLE placement into the comparison.

The runner does not start or stop vLLM. Start each profile with the intended
GPU selection and port, run the helper, then stop the service using the normal
launcher lifecycle. Keep manifests outside the source tree (for example under
`results/qwenflash/`) so model outputs and logs are not committed.

Historical measurements may be cited as context, but only a manifest produced
by this protocol is current promotion evidence. Report median and mean for the
three measured samples, plus the GPU summary captured during each request.
