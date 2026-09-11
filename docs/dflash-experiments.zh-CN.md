# DFlash 实验路线

本文档记录历史 DFlash 实验矩阵，以及当前已经晋升的 27B Qwen3.8 DFlash2
路线。

当前状态：

- 下面的 Qwen3.6 DFlash2/DFlash3 矩阵仍然是实验路线。
- Qwen3.8 NVFP4 DFlash2 已在目标双 2080 Ti runtime 上完成验证，并晋升到
  normal 正式 profile。

正式 profile 为：

- `qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env`

它使用 `nvidia/Qwen3.8-27B-NVFP4` 和 `incoai/Qwen3.8-27B-DFlash2`，采用
TurboQuant K8V4 KV、K=7、草稿侧 TRITON_ATTN、PIECEWISE CUDA Graph，以及
SM75 B=8 分块的因果 target 验证路径。固定高接受率 4K/128 三次 decode 为
155.16、166.57、166.76 tok/s。该路线正式上下文为 256K 纯文本，并通过了
正确性、接近满上下文和数千 token 输出阶段压力测试。

当前实验 profile 包括：

- `qwen27b/experimental/fp8/fp16kv-8K-dflash3-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash2-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash2-drafttp1-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-8K-dflash3-drafttp1-text-only.env`
- `qwen27b/experimental/fp8/fp16kv-128K-dflash3-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash2-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash2-drafttp1-text-only.env`
- `qwen27b/experimental/int4/fp16kv-8K-dflash3-drafttp1-text-only.env`
- `qwen27b/experimental/int4/fp16kv-256K-dflash3-text-only.env`

其中 8K 这些 profile，刻意覆盖了当前短上下文 speed sweep 的默认矩阵：

- `dflash2` + draft TP `auto`
- `dflash2` + draft TP `1`
- `dflash3` + draft TP `auto`
- `dflash3` + draft TP `1`

这些实验启动 profile 统一使用 launcher shortcut 形式
（`SPECULATIVE_METHOD=dflash` + `SPECULATIVE_TOKENS`），不要再和 `MTP_K`
或者单独的 `SPECULATIVE_CONFIG` 混用。

这些历史 profile 仍不放进正式的 `normal/` 或 `fast/` catalog，因为它们尚未具备与正式 DFlash2 路线相同的证据：

- 质量 smoke 通过，
- 在双 2080 Ti runtime 上服务能稳定拉起，
- 且短上下文速度路线的合成 decode 性能能够超过同口径生成的 MTP3 baseline。

## 前置条件

- 已验证的双 2080 Ti runtime，tensor parallel size 为 `2`。
- 一份 27B FP8 目标模型目录。
- 一份 27B INT4 目标模型目录。
- 一份匹配的 DFlash draft model，通常是 `z-lab/Qwen3.6-27B-DFlash`。

如果 draft model 不是本地路径，而是 Hugging Face repo id，launcher 现在会在
真正启动服务时自动探测 official endpoint 和 mirror endpoint，并选择可达且更快
的一条路。若需要手动固定路线，可以直接设置 `HF_ENDPOINT`，或者设置
`HF_DOWNLOAD_ROUTE_MODE=official|mirror`。

正式比较前，先用 launcher 做 dry-run：

```bash
bash launcher.sh --print-config \
  --model-dir /path/to/target-27b \
  --profile qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env \
  --gpu-devices 0,1 \
  --tp-size 2
```

摘要里应看到 `Spec decode: dflash/3 (...)`。如果 draft model 用的是 repo id，
`DFlash draft fetch` 会显示计划采用的路由策略；真正的 endpoint 选择只会在
服务实际启动时发生。

## 单个 Profile 比较

可以用服务侧 compare helper，把某个 DFlash profile 和同基础 profile
自动生成的 MTP3 baseline 做对比：

```bash
bash tools/compare_dflash_service.sh \
  --case qwen27b-int4-8k \
  --model-dir /path/to/int4-27b \
  --profile qwen27b/experimental/int4/fp16kv-8K-dflash3-text-only.env \
  --draft-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --out-dir /tmp/qwen27b-int4-dflash \
  --require-dflash-beats-mtp
```

这个 helper 会：

- 从同一个基础 profile 自动生成临时 MTP3 路线，
- 先跑 `launcher.sh --print-config`，
- 再通过 `launcher.sh --non-interactive` 拉起服务，
- 跑一轮确定性的 `PROFILE_OK` 质量 smoke，并要求 DFlash 与 MTP 基线的
  回复内容一致，
- 然后跑 warmup + synthetic completion requests，
- 最后写出 `*-compare-service.json`。

## 批量评测

可以用批量脚本把当前默认的四个 case 实验集一次跑完：

```bash
bash tools/evaluate_dflash_profiles.sh \
  --fp8-model-dir /path/to/fp8-27b \
  --int4-model-dir /path/to/int4-27b \
  --dflash-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2
```

默认 case 集合为：

- FP8 8K：`fast`，要求速度门槛通过。
- INT4 8K：`fast`，要求速度门槛通过。
- FP8 128K：`normal`，做质量观察。
- INT4 256K：`normal`，做质量观察。

批量脚本也支持把 DFlash 调优参数继续往下透传：

- `--speculative-tokens`
- `--draft-tp-size`
- `--draft-max-model-len`
- `--draft-attention-backend`
- `--disable-padded-drafter-batch`

输出包括：

- `cases.tsv`
- `summary.tsv`
- `verdict.txt`
- 每个 case 单独的 compare JSON 和日志目录
- 如果服务暴露了 Prometheus speculative-decoding counter，每个 variant
  还会额外生成一个 `spec_decode_metrics.json`

## Speed Sweep

如果需要专门找出 8K 短上下文下比 MTP 更快的 DFlash 候选，可以用 speed-sweep
helper 扫描不同的 token 数和 draft TP 配置：

```bash
bash tools/evaluate_dflash_speed_sweep.sh \
  --fp8-model-dir /path/to/fp8-27b \
  --int4-model-dir /path/to/int4-27b \
  --dflash-model /path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --speculative-tokens-list 2,3 \
  --draft-tp-size-list auto,1
```

输出包括：

- `combined-summary.tsv`：每个权重 x 每个调优候选一行
- `best.tsv`：按 decode ratio 选出的每个权重的最佳通过候选
- 每个调优候选各自的完整批量结果目录

选出胜出候选后，可以把它物化成一个实际可启动的 profile：

```bash
bash tools/materialize_dflash_profile.sh \
  --base-profile qwen27b/experimental/fp8/fp16kv-8K-dflash3-text-only.env \
  --output /tmp/fp16kv-8K-dflash2-drafttp1-text-only.env \
  --speculative-tokens 2 \
  --draft-tp-size 1
```

如果希望一次性把 `best.tsv` 里的 FP8 / INT4 最优候选全部物化出来，也可以：

```bash
python3 tools/materialize_dflash_profiles_from_best.py \
  --best-tsv /path/to/dflash_speed_sweep/best.tsv \
  --output-root /tmp/dflash-materialized
```

## 远端目标机执行

如果当前工作区不是目标 runtime，可以用远端 wrapper，把 DFlash 实验相关文件同步到
已准备好的 runtime 目录里，并在远端直接执行批量评测：

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --probe-only \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash
```

如果 probe 已确认 GPU、runtime Python 和模型路径都有效，再执行真正的批量评测：

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2
```

这个 wrapper 会：

- 支持 `--probe-only`，先检查远端 GPU 可见性、runtime 路径、模型路径和
  Python 侧 DFlash 支持情况，再决定是否真正同步和跑 benchmark，
- 支持把 SSH 登录用户（`--ssh-user`）和远端 runtime 持有用户
  （`--remote-user`）分开指定，
- 同步 `profiles/`、`tools/`、`launcher.sh`、`build.sh`、`VERSION`，以及当前本地
  `vllm/` 源码树到远端 runtime，
- 在远端以目标 runtime 用户执行 `tools/evaluate_dflash_profiles.sh`，
- 再把结果目录回传到本地工作区。

远端 wrapper 同样支持和批量脚本一致的 DFlash 调优透传参数。

如果希望直接在远端 runtime 上跑短上下文 speed sweep，可以切到：

```bash
bash tools/evaluate_dflash_profiles_remote.sh \
  --runner speed-sweep \
  --remote-host example.com \
  --ssh-user loginuser \
  --remote-root /path/to/runtime \
  --remote-user runtimeuser \
  --fp8-model-dir /remote/path/to/fp8-27b \
  --int4-model-dir /remote/path/to/int4-27b \
  --dflash-model /remote/path/to/Qwen3.6-27B-DFlash \
  --gpu-devices 0,1 \
  --tp-size 2 \
  --speculative-tokens-list 2,3 \
  --draft-tp-size-list auto,1
```

结果解释：

- `quality_gate_ok=True` 表示生成的 MTP3 路线和 DFlash 路线都返回了
  HTTP 200，通过了 `PROFILE_OK` smoke，并且确定性质量回复一致。
- `quality_text_equal_stripped=True` 表示两条路线的确定性质量回复在去掉首尾
  空白后完全一致。
- `request_gate_ok=True` 表示实测 synthetic request 都成功完成。
- `dflash_beats_mtp=True` 默认只对短上下文 8K 速度路线作为硬性要求。
- `summary.tsv` 里的 `mtp_acceptance_rate` 和 `dflash_acceptance_rate`
  可以帮助解释 decode 速度为什么赢或输。

长上下文 DFlash 路线在具备目标 runtime 上的容量证据和质量证据之前，
都不应该从 `experimental` 提升为正式 profile。
