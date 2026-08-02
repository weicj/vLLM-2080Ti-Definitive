# Huihui Qwen3.6 27B AWQ/MTP3 并发 1～4 吞吐测试

测试日期：2026-08-02  
测试服务端口：`18087`  
GPU：物理 GPU 6、7（RTX 2080 Ti / SM75，TP=2）  
模型：`/home/ubuntu/storage/llm/Qwen/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`

## 1. 测试目的

验证两件事：

1. 服务配置为 `max-num-seqs=4` 后，只发送一个请求时是否会比 `max-num-seqs=1` 明显变慢。
2. 同一服务允许最多 4 个序列时，并发 1、2、3、4 对单请求 t/s 和 aggregate t/s 的影响。

## 2. 统一配置

除 `max-num-seqs` 和 CUDA Graph capture 列表外，测试配置保持一致：

```text
MTP：3 speculative tokens
Quantization：awq_marlin
Attention：FlashInfer
GDN prefill：FlashQLA legacy SM70/SM75 prebuilt kernel
Sampler：FlashInfer sampler
CUDA Graph：PIECEWISE
custom all-reduce：关闭
max-num-batched-tokens：2048
max-model-len：245760
max-num-seqs=1 基准：cudagraph_capture_sizes=[4]
max-num-seqs=4 服务：[4,8,12,16]，max_cudagraph_capture_size=16
CPU affinity：16-31,48-63
多模态上限：{"image":4,"video":0,"audio":0}
```

请求统一为：

```text
prompt：4096 tokens
generation：128 tokens
endpoint：/v1/completions
temperature：0
ignore_eos：true
纯 filler prompt，每轮使用不同 salt，避免 prefix cache 复用造成测试偏差
每档：1 次 warmup + 3 次 measured rounds
```

本次吞吐请求是纯文本 synthetic benchmark；服务本身仍然开启了多模态，`image=4` 是允许上限，但没有把视觉编码器耗时混入文本吞吐对比。

## 3. 统计口径

### avg t/s

每个并发请求单独计算：

```text
单请求 decode t/s = completion_tokens / (该请求 elapsed - 该请求 TTFT)
```

然后对同一并发批次内的请求取算术平均，再对 3 个 measured rounds 取平均。

### aggregate decode t/s

所有并发请求产生的 completion token 总数，除以：

```text
最后一个请求结束时间 - 第一个请求收到首 token 的时间
```

这个指标主要反映 GPU 在并发 decode 阶段的总吞吐。

### aggregate end-to-end t/s

所有 completion token 总数，除以：

```text
最后一个请求结束时间 - 第一个请求开始时间
```

这个指标包含 4096-token prompt 的 prefill 和调度等待，因此会明显低于 aggregate decode t/s。

## 4. 结果汇总

| 服务配置 | 请求并发 | avg t/s（3轮均值） | avg t/s（3轮中位） | aggregate decode t/s（均值） | aggregate E2E t/s（均值） |
|---|---:|---:|---:|---:|---:|
| `max-num-seqs=1`, Graph `[4]` | 1 | **70.98** | 71.03 | **70.98** | 30.90 |
| `max-num-seqs=4`, Graph `[4,8,12,16]` | 1 | **71.28** | 70.87 | **71.28** | 30.95 |
| `max-num-seqs=4`, Graph `[4,8,12,16]` | 2 | 58.83 | 62.50 | 110.03 | 46.33 |
| `max-num-seqs=4`, Graph `[4,8,12,16]` | 3 | 62.04 | 61.94 | 170.39 | 68.30 |
| `max-num-seqs=4`, Graph `[4,8,12,16]` | 4 | 61.36 | 61.71 | **211.78** | **88.13** |

各 measured round 的原始结果：

| 并发 | avg t/s 各轮 | aggregate decode t/s 各轮 | aggregate E2E t/s 各轮 |
|---:|---|---|---|
| max-seqs=1，load=1 | 71.03 / 70.75 / 71.16 | 71.03 / 70.75 / 71.16 | 30.96 / 30.86 / 30.87 |
| max-seqs=4，load=1 | 70.87 / 72.30 / 70.67 | 70.87 / 72.30 / 70.67 | 30.99 / 31.14 / 30.71 |
| max-seqs=4，load=2 | 62.93 / 51.06 / 62.50 | 125.82 / 79.33 / 124.94 | 49.58 / 40.12 / 49.29 |
| max-seqs=4，load=3 | 61.94 / 61.57 / 62.61 | 169.99 / 169.32 / 171.86 | 68.49 / 68.10 / 68.33 |
| max-seqs=4，load=4 | 61.85 / 61.71 / 60.51 | 213.26 / 212.85 / 209.23 | 88.41 / 88.35 / 87.62 |

## 5. 结论

### 5.1 允许 4 并发不会让单并发变慢

对比最重要的两行：

```text
max-num-seqs=1，单并发：70.98 t/s
max-num-seqs=4，单并发：71.28 t/s
```

差异约 `+0.42%`，在当前机器时钟和请求抖动范围内可以认为没有性能损失。也就是说，把服务配置为允许 4 个请求，并不会自动让单用户请求从约 71 t/s 降到更低。

### 5.2 并发 2～4 能提高总吞吐，但单请求速度下降

稳定结果的趋势是：

```text
并发 1：单请求约 71 t/s，aggregate 约 71 t/s
并发 2：单请求约 59～63 t/s，aggregate decode 中位约 125 t/s
并发 3：单请求约 62 t/s，aggregate decode 约 170 t/s
并发 4：单请求约 61 t/s，aggregate decode 约 212 t/s
```

因此，如果目标是多用户总吞吐，4 并发明显有价值；如果目标是单用户最低延迟，仍然是单并发最合适。

### 5.3 并发 2 的一轮异常不是稳定性能

并发 2 的第二个 measured round 出现了一个请求的 decode t/s 降到约 `39.66`，导致该轮 avg t/s 为 `51.06`、aggregate decode 为 `79.33`。同一轮另一个请求约 `62.45 t/s`，前后轮也都约 `62.5 t/s`。

所以并发 2 同时报告均值和中位数：均值被这一轮调度抖动拉低，`62.50 t/s / 124.94 aggregate`更能代表稳定轮。

## 6. 冷启动和稳定性记录

第一次启动 `max-num-seqs=4` 后，首次并发 2 请求触发了新的 Triton Mamba/GDN shape autotune：

```text
chunk_gated_delta_rule_fwd_kernel_h_blockdim64：约 125 秒
其他 recompute / merge / chunk kernels：几十秒到 80 秒
```

由于 vLLM 默认：

```text
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
```

首次 autotune 超过 300 秒后，`sample_tokens` RPC 超时，EngineCore 被判定失败。这一轮没有有效 t/s，已排除，不作为性能结果。

重新设置：

```bash
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
```

并等待 Triton cache 建立后，max-seqs=4 服务可以稳定完成并发 2、3、4 测试，未出现 OOM 或 EngineCore 崩溃。

这说明当前 SM75 runtime 的并发配置可以运行，但首次遇到新 batch shape 时必须给足 JIT/autotune 时间。

## 7. 推荐启动方式

单请求优先的默认脚本仍然是：

```bash
cd /mnt/gw600/vllm-weicj-sm75-test
./start_huihui_qwen36_awq_mtp_sm75.sh
```

如果要开启最多 4 个并发：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
PORT=18087 \
MAX_NUM_SEQS=4 \
CG_CAPTURE_SIZES='[4,8,12,16]' \
MAX_CG_CAPTURE_SIZE=16 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
./start_huihui_qwen36_awq_mtp_sm75.sh
```

当前脚本默认允许一个请求包含最多 4 张图片：

```json
{"image":4,"video":0,"audio":0}
```

这和并发数相互独立：4 张图片是一个请求内的媒体数量上限，`MAX_NUM_SEQS=4` 才是最多 4 个请求并发。

## 8. 原始数据

- 单并发基准：[concurrency-max1.jsonl](./concurrency-max1.jsonl)
- 允许 4 并发、load=1：[concurrency-max4-c1.jsonl](./concurrency-max4-c1.jsonl)
- 允许 4 并发、load=2 稳定重测：[concurrency-max4-c2-retry.jsonl](./concurrency-max4-c2-retry.jsonl)
- 允许 4 并发、load=3：[concurrency-max4-c3.jsonl](./concurrency-max4-c3.jsonl)
- 允许 4 并发、load=4：[concurrency-max4-c4.jsonl](./concurrency-max4-c4.jsonl)
- max-seqs=4 服务日志：[concurrency-max4-retry.server.log](./concurrency-max4-retry.server.log)
- 并发测试工具：[profile_concurrency.py](../tools/profile_concurrency.py)

## 9. 扩展测试：8 并发、1024-token 长输出

在同一端口 `18087` 重新启动了 `max-num-seqs=8` 服务：

```text
max-num-seqs=8
cudagraph_capture_sizes=[4,8,12,16,24,32]
max_cudagraph_capture_size=32
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
prompt=4096 tokens
每路 generation=1024 tokens
并发=8
1 次 warmup + 5 次 measured rounds
```

8 路请求全部成功生成完整的 1024 tokens，没有 OOM、HTTP 500 或 EngineCore 退出。

| 指标 | 5轮均值 | 5轮中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 单请求 avg t/s | **56.78** | 56.21 | 52.51 | 62.40 |
| aggregate decode t/s | **396.59** | 381.14 | 359.60 | 460.18 |
| aggregate E2E t/s | **338.74** | 328.98 | 307.84 | 385.63 |

各轮 measured 结果：

| round | avg t/s | aggregate decode t/s | aggregate E2E t/s | 每请求完成 token |
|---:|---:|---:|---:|---:|
| 1 | 56.21 | 361.28 | 307.84 | 1024 |
| 2 | 52.51 | 381.14 | 328.98 | 1024 |
| 3 | 62.40 | 460.18 | 385.63 | 1024 |
| 4 | 58.00 | 420.75 | 358.04 | 1024 |
| 5 | 54.80 | 359.60 | 313.22 | 1024 |

相比之前 4 并发、128-token 短输出的约 `211.8 aggregate decode t/s`，8 并发长输出的中位数约 `381.1 aggregate decode t/s`。由于生成长度、MTP acceptance、请求结束时间和调度形状不同，两组 aggregate 数字不能简单当作同一 benchmark 的线性倍数，但可以确认 8 路长生成已经稳定达到约 `360～460 aggregate decode t/s`。

8 并发下单请求 t/s 从 4 并发的约 `61.4` 降到了约 `56.8`，同时批次间抖动更明显。因此：

- 追求总吞吐：8 并发有价值；
- 追求单请求速度和稳定延迟：4 并发更合适；
- 8 并发首次启动仍建议保留 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900`，避免 SM75 新 shape autotune 超时。

8 并发原始数据：

- [8并发1024-token主测试](./concurrency-max8-c8-gen1024.jsonl)
- [8并发1024-token补充测试](./concurrency-max8-c8-gen1024-extra.jsonl)
- [max-seqs=8服务日志](./concurrency-max8.server.log)

## 10. 固定 `max-num-seqs=8` 下的实际并发 1～8 扫描

上一节只回答了“8-seq 服务跑 8 个请求”的结果。本节固定同一个服务实例：

```text
max-num-seqs=8
cudagraph_capture_sizes=[4,8,12,16,24,32]
prompt=4096 tokens
每个请求生成=1024 tokens
实际请求并发=1、2、3、4、5、6、7、8
每档 1 次 warmup + 3 次 measured rounds
```

这里的“并发 1”不是重新启动 `max-num-seqs=1` 服务，而是明确在已经允许 8 路的同一个服务上只提交 1 个请求。这正是检验开启 8 并发后单路是否变慢的结果。

| 实际并发 | avg t/s 均值 | avg t/s 中位数 | aggregate decode t/s 均值 | aggregate decode t/s 中位数 | aggregate E2E t/s 均值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 65.22 | 67.24 | 65.22 | 67.24 | 56.57 |
| 2 | 68.96 | 69.50 | 137.92 | 138.99 | 113.50 |
| 3 | 67.28 | 67.29 | 199.40 | 199.26 | 162.90 |
| 4 | 57.34 | 56.18 | 204.72 | 197.88 | 174.65 |
| 5 | 55.98 | 55.24 | 249.16 | 247.71 | 213.53 |
| 6 | 54.75 | 51.97 | 295.37 | 284.87 | 253.18 |
| 7 | 58.16 | 57.75 | 374.80 | 377.45 | 317.82 |
| 8 | 56.91 | 55.33 | **429.19** | **413.59** | **363.58** |

### 10.1 结论

1. 在同一个 `max-num-seqs=8` 服务中只跑 1 个请求，avg t/s 均值约 `65.2`，没有因为预留 8 路 Graph 而直接掉到很低。
2. 实际并发 2～3 时，单请求仍约 `67 t/s`；并发 4～6 时，单请求下降到约 `55～57 t/s`，但 aggregate 吞吐继续增加。
3. 并发 8 的本组 3 轮均值约 `429 aggregate decode t/s`，中位数约 `414 aggregate decode t/s`，每个请求约 `57 t/s`。
4. 曲线不是严格线性：并发 4 的 aggregate 只有约 `205 t/s`，而并发 5、6、7、8 分别约 `249/295/375/429 t/s`。这说明 vLLM scheduler、MTP acceptance 和不同 Graph shape 的组合会产生批次抖动，不能用“并发数×单请求 t/s”直接估算。
5. 本节的 c8 数值与上一节单独 c8 的 5 轮均值（约 `396.6 t/s`）不同，原因是两次独立服务运行的 decode 调度抖动；两组都完整生成 1024 tokens，均没有错误。做 1～8 曲线时，应以本节同一服务实例的数据为准。

本节原始数据：

- [实际并发1](data/concurrency-max8-load1-gen1024.jsonl)
- [实际并发2](data/concurrency-max8-load2-gen1024.jsonl)
- [实际并发3](data/concurrency-max8-load3-gen1024.jsonl)
- [实际并发4](data/concurrency-max8-load4-gen1024.jsonl)
- [实际并发5](data/concurrency-max8-load5-gen1024.jsonl)
- [实际并发6](data/concurrency-max8-load6-gen1024.jsonl)
- [实际并发7](data/concurrency-max8-load7-gen1024.jsonl)
- [实际并发8](data/concurrency-max8-load8-gen1024.jsonl)
- 各档补测文件：`concurrency-max8-load{1..8}-gen1024-extra.jsonl`
- [固定8-seq服务日志](./concurrency-max8-1to8.server.log)
