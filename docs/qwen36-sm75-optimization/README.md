# Qwen3.6-27B AWQ on RTX 2080 Ti (SM75)

这是一份可公开复核的本地优化实验包，记录 `weicj/vLLM-2080Ti-Definitive` 思路在
双 RTX 2080 Ti（SM75）上的 Qwen3.6-27B AWQ 多模态运行结果。测试使用 GPU 6、7，
TP=2，保留 CUDA Graph，并保留 `image=4` 的多模态上限。

## 先看结果

严格同口径的真实长输出（4096 prompt + 1024 generated tokens，SSE token 时间戳）：

| profile | decode | TTFT | 说明 |
|---|---:|---:|---|
| baseline | 25.53 tok/s | 4.82 s | no MTP、Triton attention、native sampler、PIECEWISE Graph `[1]` |
| optimized | 49.94 tok/s | 3.02 s | MTP3、FlashQLA legacy GDN、FlashInfer attention/sampler、PIECEWISE Graph `[4]` |

这是一轮同 prompt 的热缓存长测，提升约 **1.96x**。此前独立的 4096/128 短测为
`38.77 -> 74.67 tok/s`；短测峰值不能当作 1024-token 长稳态吞吐，因此视频把两种
口径明确分开。基线长测还记录过 30.39 tok/s，优化长测记录过 42.87、49.29 tok/s，
这些波动也保留在本地 `test-results/` 中，不能只挑最好的一次宣称固定速度。

视频：[qwen36-sm75-throughput-before-after-20260802.mp4](video/qwen36-sm75-throughput-before-after-20260802.mp4)

视频的每个进度点来自 `/v1/completions` 的真实 SSE 到达时间；没有把短测数据插值成
长测，也没有伪造 token 事件。视频编码为 H.264、1280×720、30 fps。

## 推荐运行配置

日常交互和最多四路并发优先使用 `max-num-seqs=4`；严格 1024-token 复测中，四路时
seq=4 的 aggregate decode 为 232.76 tok/s，seq=8 为 204.72 tok/s。只有确实需要
5～8 路并发时才使用 seq=8；seq=8 的八路 aggregate 约 429.19 tok/s，但单请求降到
约 56.91 tok/s。

```bash
CUDA_VISIBLE_DEVICES=6,7 \
MAX_NUM_SEQS=4 \
CG_CAPTURE_SIZES='[4,8,12,16]' \
MAX_CG_CAPTURE_SIZE=16 \
./start_huihui_qwen36_awq_mtp_sm75.sh
```

图像数量和并发序列是两个独立维度：`--limit-mm-per-prompt '{"image":4,"video":0,"audio":0}'`
表示单个请求最多 4 张图，不表示只能有 4 个并发请求。

## 优化栈

- AWQ Marlin W4A16，FP16 KV，TP=2。
- `MTP3`，并使用 `VLLM_SM75_SPEC_SYNC_MODE=safe`。
- FlashQLA SM75 legacy GDN prefill。
- FlashInfer attention 与 FlashInfer sampler（用隔离的 0.6.8 overlay）。
- `cudagraph_mode=PIECEWISE`，视频 profile 的 capture size 为 `[4]`；没有 `--enforce-eager`。
- `image=4` 保留，多模态服务启动路径不关闭。
- 当前稳定 profile 显式关闭 custom all-reduce。已有同模型对照显示，打开它文本慢约
  2.25%、图片慢约 3.00%，且在 SM75 上没有稳定收益；它可以作为单独实验开关，不是默认快路径。

视频复现实验服务脚本在 `scripts/start_video_server.sh`，默认端口为 `18087`，默认测试卡为
`6,7`。`MODE=baseline` 和 `MODE=optimized` 分别启动两套 profile；长测请求由
`scripts/record_stream_timeline.py` 记录，视频由 `scripts/make_throughput_video.py` 渲染。

## 报告和原始记录

- [完整模型/runtime 报告](experiment-report-20260802.md)
- [seq=4 与 seq=8 合并报告（包含两者都生成 1024 tokens 的严格复测）](concurrency-seq4-vs-seq8-merged-20260802.md)
- [并发 1～4/1～8 原始报告](concurrency-1-4-report-20260802.md)
- `data/`：严格长输出并发 JSONL、项目口径 4096/128 对照 JSONL。
- `video/`：真实长输出 before/after 视频。

## 可复现边界

这不是 README 中某个固定数字的硬件保证。公开项目数字还依赖 checkpoint、CUDA/torch/
FlashInfer 版本、CPU 单核频率和 TP 控制面；本实验使用的是本机 Huihui AutoRound-AWQ
checkpoint、当前 weicj runtime 和 CUDA 13.0 torch 进程。GPU/模型/参数相同并不等于运行时
完全相同。尤其不要把 `94 tok/s` 的其他 checkpoint 项目数字直接套到本模型的 1024-token
长输出。

原始模型权重、编译缓存、服务器日志和本机绝对路径没有提交；仓库只保留可公开的报告、
压缩后的结果记录、复现实验脚本和视频。
