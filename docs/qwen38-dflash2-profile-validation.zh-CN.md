# Qwen3.8 DFlash2 Profile 验证记录

这是
`profiles/qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env` 的复现记录。

## 路线

- 目标模型：`nvidia/Qwen3.8-27B-NVFP4`
- 草稿模型：`incoai/Qwen3.8-27B-DFlash2`
- 硬件：双 RTX 2080 Ti（SM75），启用 P2P，custom all-reduce 为 auto
- KV：TurboQuant K8V4
- 投机：DFlash2，K=7，`TRITON_ATTN`
- 目标图：`PIECEWISE`，capture size 8
- TQ target 验证：因果 synthetic sequence length，B=8 分块
- 上下文：`MAX_MODEL_LEN=262144`
- 范围：纯文本，单序列

## 证据

固定高接受率 4K prompt / 128 completion 三次独立测试的 decode 速度为：

```text
155.164865 tok/s
166.566925 tok/s
166.756140 tok/s
平均 162.829310 tok/s
```

三次均返回 128/128 token、HTTP 200 且 stream 正常结束。正确性 smoke 三次
都返回 `PROFILE_OK`。接近满长的 `262016` token prompt 完成了 16/16 输出。
真实输出阶段压力测试生成了约 4K 和 8K token 的 HTML/Canvas 交互任务；后者
包含 JavaScript，并在达到长度上限前保持服务健康。服务进程持续存活，日志没有
新的 CUDA illegal instruction、`EngineDead`、fatal error 或 stream 中断。

目标 runtime 上的 DFlash2 单元测试通过：

```text
17 passed in 10.84s
```

## 复现

使用 normal launcher 路线：

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
PROFILE=qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env \
GPU_DEVICES=GPU-UUID-0,GPU-UUID-1 TP_SIZE=2 MODE=normal \
bash launcher.sh --non-interactive --print-config
```

最终环境必须包含 `SPECULATIVE_METHOD=dflash`、`SPECULATIVE_TOKENS=7`、
`SPECULATIVE_MAX_MODEL_LEN=262144`、
`SPECULATIVE_ATTENTION_BACKEND=TRITON_ATTN`、
`VLLM_TURBOQUANT_CUDAGRAPH_SPEC_DECODE_SAFE=1` 和
`VLLM_TURBOQUANT_SPEC_DECODE_CHUNK_SIZE=8`。

B=8 是 SM75 性能调优参数，不改变语义：每一行 target verification 仍然使用
递增的因果 sequence length。使用该 profile 时不要关闭 safe branch。
