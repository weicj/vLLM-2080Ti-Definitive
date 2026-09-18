# 8xT10 Profile

语言：[English](README.md) | 简体中文

## Qwen3.8 Flash-Next NVFP4（实验性 PP）

以下路线面向八张 Tesla T10（`0,2,3,4,6,7,8,9`），使用 ModelOpt NVFP4
权重的 SM75 安全 Marlin W4A16 路径。路线固定 FP8 KV（当前被 Turing KV allocator 阻塞）、`MTP_K=0`、
`flashqla_legacy`、NCCL collective、同步调度，并显式设置 `PLE_PLACEMENT=disk`（safetensors
mmap/page-cache lookup）。`cpu` 使用 pinned host memory + UVA，`gpu` 则将表常驻显存。

| Profile | TP/PP | 上下文 | PLE | 4K/128 | 32K/512 |
|---|---:|---:|---|---|---|
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 4x2 | 32K | disk | 阻塞 | 阻塞 |
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 2x4 | 32K | disk | 阻塞 | 阻塞 |

历史 `.31` 基线（不作为当前晋升证据）：早期 QwenFlash PP 验证记录了 TP4xPP2 的
4K/128 性能 `1615.30 / 21.99 tok/s`、GPU KV tokens `121,139`，以及 TP2xPP4 的
`1979.79 / 10.74 tok/s`、`131,872` GPU KV tokens。另有一组高容量 TP4xPP2 记录为
`1697.99 / 28.18 tok/s`、`100,031` GPU KV tokens。这些记录使用审计协议之前的
runtime/profile，不能证明当前分支的 FP8-KV 状态，也没有形成 32K/512 结果。

先启动服务，再从仓库根目录运行审计 runner。它会对每种 shape 执行一次预热和三次
正式采样，并将 commit、profile、环境、GPU、请求契约和逐样本数据写入 JSON manifest：

```bash
python3 tools/benchmark_qwenflash_audit.py \
  --model-dir /path/to/qwen38-flash-next-nvfp4 \
  --served-name qwen38-flash-next-nvfp4-tp4pp2 \
  --base-url http://127.0.0.1:8000/v1 \
  --python /path/to/runtime/.venv/bin/python \
  --profile profiles/8xT10/qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env \
  --output results/qwenflash/tp4pp2-audit.json
```

runner 会拒绝未完成的 stream，以及 prompt 或 completion token 数不符合 `4K/128`、
`32K/512` 契约的请求。manifest schema 和证据口径见[审计协议](../../docs/qwenflash-audit.zh-CN.md)。
