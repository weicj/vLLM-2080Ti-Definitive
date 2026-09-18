# Qwen Flash-Next 审计协议

`tools/benchmark_qwenflash_audit.py` 在不接管服务进程的前提下测量运行中的
Qwen3.8 Flash-Next NVFP4 服务。每种 shape 执行一次预热，再执行三次正式请求：

| Shape | Prompt | Completion | 样本 |
|---|---:|---:|---:|
| `4k128` | 4096 | 128 | 1 次预热 + 3 次正式 |
| `32k512` | 32768 | 512 | 1 次预热 + 3 次正式 |

helper 使用 completions endpoint、纯 filler prompt 和单个允许 token，并校验精确的
prompt/completion token 数、HTTP 200、stream 完成以及正数的 prefill/decode 速率。
任何校验失败都不会产生有效审计结果。

如果 benchmark 依赖安装在运行时 venv，请使用
`--python /path/to/runtime/.venv/bin/python`；默认使用启动审计脚本的解释器。

JSON manifest 记录 UTC 时间段、相对 profile 和所有路线 key、模型路径、服务 URL、
git commit/branch/dirty 状态、GPU 清单、benchmark 契约、实际 KV cache dtype，以及
每个 `profile_request.py` 原始记录。profile 可以使用 `float16` 或 `fp8` KV，但
dtype 是结果身份的一部分，不能在不注明差异的情况下直接比较。profile 仍必须声明
`MTP_K=0`、`SPECULATIVE_METHOD=none` 和 `PLE_PLACEMENT=disk`，避免把 MTP、EXL3
或未记录的 PLE placement 混入对比。

runner 不会启动或停止 vLLM。请按目标 GPU 和端口用正常 launcher 生命周期启动每个
profile，运行 helper，再停止服务。manifest 建议写到源码树外（例如
`results/qwenflash/`），不要把模型输出和日志提交进仓库。

历史测量可以作为背景引用，但只有本协议生成的 manifest 才是当前晋升证据。报告三次
正式采样的均值和中位数，并附每次请求期间捕获的 GPU 摘要。

## 2026-09-15 证据边界

`.31` 上 2026-09-15 的记录属于另一组实验：runtime 为
`v0.29.1rc0+33.gb23433088b`，TP=2，`max_model_len=196608`，FP8 KV，GPU KV
容量为 565,438 tokens。它有成功的 HTTP 请求，但没有请求侧 timing manifest，
因此不能证明记忆中的 TP4xPP2「prefill 2000+、decode 30+」。更早的
2026-08-30 TP4xPP2 启动使用 `v0.27.1`、`max_model_len=8192`，并明确记录
SM75 FlashQLA 扩展不可用、回落到 Triton/FLA；它也不是目标结果。保留这些
记录是为了避免把不同 runtime 的结果错误归因到当前 profile。
