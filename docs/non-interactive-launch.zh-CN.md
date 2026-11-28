# 非交互启动

先执行 `./build.sh`，再传入扁平化的 profile 路径。Profile 只包含路线参数；检查点、GPU 拓扑、端口和 mode 都由 launcher 管理。

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-INT4 \
  --profile 2xT10/qwen38-27b/w4a16/mtp3-fp8kv-1x262K-text-only.env \
  --mode fast \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --pp-size 1 \
  --print-config
```

`--mode` 可省略，默认是 `fast`。没有 `MODE` 的 profile 会保留 launcher 的选择；profile 中显式的 `MODE=normal` 或 `MODE=fast` 可以覆盖它。目录中不再区分 `fast/` 和 `normal/`。

可以使用 `--model-dir`、`--speculative-model`、`--profile`、`--mode`、`--gpu-devices`、`--tp-size`、`--pp-size`、`--port`、`--start-timeout` 和 `--print-config`。高级 launcher/runtime 参数使用 `--set KEY=VALUE`。不要把 Prefix Cache、Mamba cache、GPU、端口或模型路径写入 profile，验证器会拒绝这些字段。

Profile 库是验证矩阵，不代表每个文件在每台机器上都能运行。只有外部审计完成启动、4K/128、32K/512、并发和图文正确性验证后，路线才会被 promote。

浏览器客户端与 API 来源不同时，启动服务时用 `API_ALLOWED_ORIGINS` 设置允许访问的来源列表。比如 LLM Speed Chat 可以加 `--set 'API_ALLOWED_ORIGINS=["https://weicj.github.io"]'`。每个来源填写协议、主机和可选端口，省略 URL 路径。浏览器提示访问本地网络时，请允许该网页访问。

## 实验性 SSD 前缀缓存

Launcher 可以启用 vLLM 内置的 `OffloadingConnector`，通过 CPU 暂存层和文件系统层将已完成的前缀 KV 块写入磁盘，不需要常驻的独立缓存进程。目录必须位于持久化存储上；重启后保持模型路径、KV 精度、块布局和并行配置一致。CPU 暂存空间占用主机内存，不会增加 GPU KV 容量。每次启动前，launcher 会读取目标模型和本地草稿模型的全部文件，按内容摘要选择独立的磁盘命名空间。因此在原路径替换权重后不会读到旧 KV；代价是启动时额外完整读取一次检查点。旧命名空间需要手动清理以回收空间。切换到不兼容的运行时仍需清理缓存。磁盘缓存要求检查点位于本地目录；launcher 启动和服务运行期间不要修改检查点文件。

文件系统层不会自动淘汰旧文件，也没有磁盘配额；请使用空间充足的专用卷，监控用量，并在服务停止后清理过期缓存。`KV_DISK_CPU_BYTES` 只限制主机内存暂存空间。启用磁盘缓存时 launcher 固定 `PYTHONHASHSEED=0`，并拒绝其他值，以保证重启后的块哈希稳定。

```bash
./launcher.sh \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4 \
  --gpu-devices 0,1,2,3 --tp-size 4 \
  --set KV_DISK_CACHE_DIR=/path/to/ssd-cache \
  --set KV_DISK_CPU_BYTES=4294967296 \
  --print-config
```

去掉 `--print-config` 即可启动。启用后 launcher 会要求打开前缀缓存，并设置稳定的 `PYTHONHASHSEED`。停止服务前应等待 `vllm:kv_offload_tiering_active_cascade_jobs` 回到零。重启后可查看 `vllm:external_prefix_cache_hits_total`，以及文件系统层的 `vllm:kv_offload_tiering_chunk_hits_total`、`vllm:kv_offload_tiering_read_bytes_total`。
