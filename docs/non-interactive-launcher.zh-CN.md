# 非交互 Launcher

`launcher.sh` 现在支持完整的非交互启动路径，可用于脚本部署、CI 校验和可复现的
route 验证。

最终优先级为：

`CLI > ENV > PROFILE > default`

其中 `CLI` 包括直接参数（如 `--max-model-len 262144`）、`--set KEY=VALUE`
以及 `--unset KEY`。

## 参数命名

已纳入 launcher 管理面的配置，可以直接写成 `--lower-kebab-case`：

- `MODEL_DIR` -> `--model-dir`
- `PROFILE` -> `--profile`
- `MAX_MODEL_LEN` -> `--max-model-len`
- `MM_LIMIT_JSON` -> `--mm-limit-json`

对于不在直接 launcher 参数面上的高级环境变量，使用：

- `--set KEY=VALUE`
- `--unset KEY`

`--unset KEY` 会清掉继承自 profile 或环境变量的值，然后回落到更低优先级的
launcher 默认层。如果你需要保留“显式空字符串”而不是回落默认值，使用
`--set KEY=`。

## 常见示例

只打印最终启动摘要，不真正启动服务：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env
```

基于仓库内置 profile 启动，并只覆写少量字段：

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --mode normal \
  --service-scope lan \
  --max-model-len 196608 \
  --gpu-devices 0,1
```

环境变量仍然可用，但 CLI 优先级更高：

```bash
MAX_MODEL_LEN=131072 \
MODE=fast \
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --max-model-len 262144
```

用 `--set` 传递高级运行时环境变量：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_TURBOQUANT_FLASHINFER_BACKEND=fa2 \
  --set CC=/usr/bin/gcc-12
```

清掉 profile 提供的字段，并让 launcher 回填更低层默认值：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --unset max-model-len
```

强制保留显式空值：

```bash
./launcher.sh --print-config \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set REASONING_PARSER=
```

## API 密钥认证

vLLM 的 OpenAI 兼容服务可以对 `/v1/*` 请求要求 bearer token。Launcher 通过环境变量
传递密钥，因此它不会写入 `run-logs/` 里记录的启动命令，也不会出现在服务进程参数中：

```bash
./launcher.sh --non-interactive \
  --model-dir /models/Qwen3-30B-A3B-Instruct-2507-AWQ \
  --profile qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
  --set VLLM_API_KEY=<密钥>
```

环境里存在 `VLLM_API_KEY` 时，启动 smoke test 会带上
`Authorization: Bearer $VLLM_API_KEY`，因此启用认证的启动仍能通过 smoke 校验；
环境里没有该变量时，smoke test 与服务行为完全不变。

`VLLM_API_KEY` 属于全局/高级运行时 env，不是路由 profile 字段，所以不要写进
`profiles/*.env`。认证只覆盖 `/v1/*`：`/health`、`/metrics`、`/version`、`/load`、
`/tokenize`、`/detokenize` 仍然开放，其中 `/health` 开放正是就绪探测所依赖的。
对外暴露的服务绝不要设置 `VLLM_SERVER_DEV_MODE=1`，它会挂载 `/sleep`、
`/wake_up`、`/reset_prefix_cache`、`/collective_rpc` 等免认证控制端点。

## 派生默认值

profile 加载后，launcher 仍会对部分字段做规范化：

- `MESSAGE_TYPE`、`MM_LIMIT_JSON`、`LANGUAGE_MODEL_ONLY`、
  `SKIP_MM_PROFILING`
- 模式派生运行时参数，如 `ENFORCE_EAGER`、
  `VLLM_SM75_SPEC_SYNC_MODE`、`VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH`
- 已验证 Qwen3 路线上默认补齐的 `REASONING_PARSER=qwen3`
- Qwen prefix-cache 路线默认补齐的 `MAMBA_CACHE_MODE=align`

这些默认值只会补空缺字段，不会再反向覆盖显式 CLI 输入。

## 推荐校验方式

调整脚本或部署自动化时，优先先跑 `--print-config`。它会打印最终生效的启动摘要，
并在真正启动 vLLM 前退出，是确认每个覆写项是否真正落到最终配置上的最稳妥方式。
