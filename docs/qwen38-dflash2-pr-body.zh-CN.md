# DFlash2：Qwen3.8 NVFP4 SM75 Profile

## 摘要

为双 RTX 2080 Ti SM75 晋升首条已验证的 DFlash2 路线：

- 目标模型：`nvidia/Qwen3.8-27B-NVFP4`
- 草稿模型：`incoai/Qwen3.8-27B-DFlash2`
- KV：TurboQuant K8V4
- 投机 K：7
- 目标图：PIECEWISE，capture size 8
- 草稿 attention：`TRITON_ATTN`
- 上下文：262144 token，纯文本

SM75 TurboQuant target 路径现在为每一个 proposed token 保留因果验证的
sequence length，并以 B=8 安全分块执行。这样可以保留验证语义，同时使用
CUDA Graph 路径。

## Profile

```text
profiles/qwen27b/normal/nvfp4/tqk8v4-256K-dflash2-text-only.env
```

Profile 不保存主机相关的 GPU、端口、chat template 或 reasoning 配置；这些仍由
launcher/service 设置。

## 验证

高接受率固定 4K/128 测试：

```text
155.164865 tok/s
166.566925 tok/s
166.756140 tok/s
平均：162.829310 tok/s
```

其他证据：

- 正确性 smoke 3/3 返回 `PROFILE_OK`。
- 速度 stream 3/3 返回 128/128 token 且 HTTP 200。
- 262016-token prompt 完成 16/16 输出 token。
- 真实 4K 和 8K 输出阶段 HTML/JavaScript 任务均完成，没有 stream 中断；
  8K 结果达到请求长度上限，并包含 Canvas/JavaScript 代码。
- 服务持续存活，没有新的 CUDA illegal instruction、EngineDead、fatal error
  或 stream 失败。
- 目标 runtime 的 DFlash2 单元测试：`17 passed in 10.84s`。

完整验证记录见
[`qwen38-dflash2-profile-validation.zh-CN.md`](qwen38-dflash2-profile-validation.zh-CN.md)。

## 验证命令

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile vllm/envs.py \
  vllm/v1/attention/backends/turboquant_attn.py
git diff --check
```

目标 runtime：

```bash
python -m pytest -q tests/v1/spec_decode/test_dflash2.py
```

## 范围说明

当前仓库工作树还包含用户在其他 runtime 和模型路线上的未提交改动。本 PR 说明
只界定 DFlash2/profile 这一部分；最终 git commit 或外部 PR 时应继续保持其他改动
分离。
