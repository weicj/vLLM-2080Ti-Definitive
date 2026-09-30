# 非交互启动

先执行 `./build.sh`，再传入扁平化的 profile 路径。Profile 只包含路线参数；检查点、GPU 拓扑、端口和 mode 都由 launcher 管理。

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast \
  --gpu-devices 1,5 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` 可省略，默认是 `fast`。没有 `MODE` 的 profile 会保留 launcher 的选择；profile 中显式的 `MODE=normal` 或 `MODE=fast` 可以覆盖它。目录中不再区分 `fast/` 和 `normal/`。

可以使用 `--model-dir`、`--speculative-model`、`--profile`、`--mode`、`--gpu-devices`、`--tp-size`、`--pp-size`、`--port`、`--start-timeout` 和 `--print-config`。高级 launcher/runtime 参数使用 `--set KEY=VALUE`。不要把 Prefix Cache、Mamba cache、GPU、端口或模型路径写入 profile，验证器会拒绝这些字段。

Profile 库是验证矩阵，不代表每个文件在每台机器上都能运行。只有外部审计完成启动、4K/128、32K/512、并发和图文正确性验证后，路线才会被 promote。

## 实验性外部 KV 存储

launcher 可以在本地 prefix cache 之外，显式启用 `MooncakeStoreConnector`。
Mooncake 的 metadata server、master server 和存储池需要预先部署。在运行时
venv 中安装与 CUDA 版本匹配的 Mooncake transfer-engine 包，并准备包含至少
`metadata_server` 和 `master_server_address` 的 JSON 配置。还应按实际部署设置
`mode`、`protocol`、`device_name`、`global_segment_size` 和
`local_buffer_size`。`embedded` 模式使用进程内存，重启 vLLM 后不会保留缓存。
外部服务的部署方式以 Mooncake 文档为准。

外部存储是服务设置，不属于 route profile。例如：

```bash
./launcher.sh \
  --model-dir /mnt/models/Qwen3.8-27B-FP8 \
  --profile 2x2080Ti/qwen27b/w8a16/mtp4-fp16kv-1x148K-text-only.env \
  --mode fast --gpu-devices 1,5 --tp-size 2 --pp-size 1 \
  --kv-store-backend mooncake \
  --mooncake-config-path /path/to/mooncake-store.json \
  --print-config
```

删除 `--print-config` 才会真正启动服务。launcher 在预览时检查 JSON 格式和必要地址，实际
启动前检查 `import mooncake.store`。此功能仍为实验性 connector 路线，不是
已经验证的 SM75 profile；正式使用前需验证缓存命中、输出正确性和性能。
用于 P/D 分离的 `MooncakeConnector` 是另一项独立功能。
