# `max-num-seqs=4` vs `max-num-seqs=8` 合并对比

## 1. 先给结论

如果你的实际需求是单用户交互或最多 4 个并发请求，推荐 `max-num-seqs=4`。

如果确实会同时有 5～8 个请求，推荐 `max-num-seqs=8`；它的 aggregate decode 吞吐可以继续上升到约 `429 t/s`，但单请求会下降到约 `57 t/s`，并且调度抖动更明显。

`image=4` 和 `max-num-seqs` 是两个独立参数。一个请求里放 4 张图不需要把 `max-num-seqs` 从 4 提高到 8。

## 2. 两组测试口径

共同条件：

```text
同一 Huihui-Qwen3.6-27B-AWQ-MTP
同一物理 GPU 6、7，TP=2
同一 SM75 FlashQLA + FlashInfer + AWQ Marlin runtime
同一 4096-token prompt
同一 MTP3、PIECEWISE CUDA Graph、custom all-reduce关闭
同一 image=4 多模态上限（本次吞吐请求为纯文本）
```

差异：

| 项目 | seq=4 测试 | seq=8 测试 |
|---|---|---|
| 最大 scheduler 序列 | 4 | 8 |
| Graph capture | `[4,8,12,16]` | `[4,8,12,16,24,32]` |
| 每路 generation | 128 tokens | 1024 tokens |
| 实际并发 | 1～4 | 1～8 |
| 每档 measured rounds | 3 | 3 |

因此，下面的数值可以用于容量和趋势判断，但不能把 seq=4 与 seq=8 的每一个 t/s 差异都解释成纯粹的 `max-num-seqs` 差异；generation 长度不同会改变 MTP、调度和 batch shape。

## 3. 重叠并发 1～4 对比

| 实际并发 | seq=4 avg t/s | seq=4 aggregate decode | seq=8 avg t/s | seq=8 aggregate decode |
|---:|---:|---:|---:|---:|
| 1 | **71.28** | 71.28 | 65.22 | 65.22 |
| 2 | 58.83 | 110.03 | **68.96** | **137.92** |
| 3 | 62.04 | 170.39 | **67.28** | **199.40** |
| 4 | **61.35** | **211.78** | 57.34 | 204.72 |

### 解读

- 并发 1：seq=4 的稳定性和单请求速度更好，约 71 t/s。
- 并发 2～3：seq=8 的本次长输出 aggregate 更高，但这不是严格同 generation 长度 A/B。
- 并发 4：两者 aggregate 几乎相同，seq=4 约 211.8，seq=8 约 204.7；seq=8 没有给 4 并发带来明确收益。

这组重叠数据支持：如果上限就是 4 并发，seq=4 更合适，seq=8 的额外 Graph shape 和调度容量没有转化成更高的 4 路吞吐。

## 4. seq=8 扩展到 5～8 并发

| 实际并发 | avg t/s | aggregate decode t/s | aggregate E2E t/s |
|---:|---:|---:|---:|
| 5 | 55.98 | 249.16 | 213.53 |
| 6 | 54.75 | 295.37 | 253.18 |
| 7 | 58.16 | 374.80 | 317.82 |
| 8 | 56.91 | **429.19** | 363.58 |

这才是 seq=8 的主要价值：实际请求超过 4 个时，总吞吐继续增加。代价是单请求约从 61 t/s 级别降到 55～57 t/s，且批次间抖动更明显。

## 5. 推荐配置

### 日常交互、单用户或最多 4 个请求

```bash
MAX_NUM_SEQS=4 \
CG_CAPTURE_SIZES='[4,8,12,16]' \
MAX_CG_CAPTURE_SIZE=16 \
./start_huihui_qwen36_awq_mtp_sm75.sh
```

这是更推荐的默认生产配置：Graph 少、显存工作区更小、单路和 4 路的行为更稳定。`MM_LIMIT_JSON` 仍然可以保持：

```json
{"image":4,"video":0,"audio":0}
```

### 确实需要 5～8 路并发

```bash
PORT=18087 \
MAX_NUM_SEQS=8 \
CG_CAPTURE_SIZES='[4,8,12,16,24,32]' \
MAX_CG_CAPTURE_SIZE=32 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
./start_huihui_qwen36_awq_mtp_sm75.sh
```

首次启动 seq=8 建议提高执行超时，因为 SM75 新 Graph/Mamba shape 的 Triton autotune 可能持续数分钟；缓存建立后，推理阶段已经实测可以稳定完成 1～8 并发长输出。

## 6. 原始报告

- [seq=4 与并发1～4报告](./concurrency-1-4-report-20260802.md)
- [seq=8 与实际并发1～8报告](./concurrency-1-4-report-20260802.md#10-固定-max-num-seqs8-下的实际并发-18-扫描)
- seq=4 原始文件见 [`data/`](data/)，本公开包保留严格长输出的四档 JSONL。
- seq=8 原始文件：`data/concurrency-max8-load{1..8}-gen1024.jsonl`（本公开包不提交服务器日志）。

## 7. 严格同口径复测：两者都生成 1024 tokens

为最终决定 seq 值，又重新启动了 `max-num-seqs=4` 服务，并使用与 seq=8 完全相同的长输出条件：

```text
prompt=4096 tokens
每路 generation=1024 tokens
actual concurrency=1、2、3、4
seq=4 Graph=[4,8,12,16]
seq=8 Graph=[4,8,12,16,24,32]
每档 1 warmup + 3 measured rounds
```

| 实际并发 | seq=4 avg t/s | seq=4 aggregate decode | seq=8 avg t/s | seq=8 aggregate decode |
|---:|---:|---:|---:|---:|
| 1 | **68.60** | **68.60** | 65.22 | 65.22 |
| 2 | 67.26 | 126.57 | **68.96** | **137.92** |
| 3 | 65.36 | 194.03 | **67.28** | **199.40** |
| 4 | **63.51** | **232.76** | 57.34 | 204.72 |

这次是相同 prompt、相同生成长度、相同 GPU 和相同 runtime 的可比数据。结论非常明确：

- 并发 1：seq=4 单请求约快 `5%`；
- 并发 2～3：seq=8 aggregate 略高，但优势只有约 `3～9%`；
- 并发 4：seq=4 单请求约快 `11%`，aggregate 约高 `13.7%`；
- 因此，实际并发上限为 4 时，`max-num-seqs=4` 是更好的配置；seq=8 不会给 4 路场景带来收益。

seq=8 只有在实际并发超过 4 时才体现优势：同一组测试中并发 5、6、7、8 的 aggregate decode 均值约为 `249/295/375/429 t/s`。

严格同口径 seq=4 原始数据：

- [seq=4长输出并发1](data/concurrency-max4-long1024-load1.jsonl)
- [seq=4长输出并发2](data/concurrency-max4-long1024-load2.jsonl)
- [seq=4长输出并发3](data/concurrency-max4-long1024-load3.jsonl)
- [seq=4长输出并发4](data/concurrency-max4-long1024-load4.jsonl)
- seq=4 服务日志未公开提交；启动命令见 `scripts/start_video_server.sh`。
