# MODIFICATION_LOG —— vllm-2080Ti-dev 本地修改记录(2026-08-10)

> 纪律:改 vLLM 共享代码必须 GGUF+AWQ 双兼容 + [FORK 兼容] 注释 + 本日志登记。

## 本次会话修改

### 3. fused_recurrent.py + gdn_linear_attn.py — packed decode 内核 v-head 映射修复(2026-08-10,12*8=9 根因)✅
- 文件:vllm/model_executor/layers/fla/ops/fused_recurrent.py + layers/mamba/gdn_linear_attn.py
- 问题:`fused_recurrent_gated_delta_rule_packed_decode_kernel` 的 v-head→k-head 映射写死
  `i_h = i_hv % H`(llama.cpp GGUF/mod16 交错映射)。div3 布局(AWQ/transformers 官方)应为
  `i_h = i_hv // (HV // H)`(q/k repeat_interleave 连续分组,transformers
  modeling_qwen3_5.py:523-525 实证)。映射错乱 → decode 阶段部分 v-head 输出反相
  (探针实测 head 2/7/15/22 余弦 -1.0)→ 生成错乱(12*8=9、23*47=2347)
- 修复 3 处([FORK 兼容 2026-08-10 GPTQ8]):
  1. packed decode 内核加 `GGUF_LAYOUT: tl.constexpr` 分支:GGUF 保持 `% H`,div3 用 `// (HV // H)`
  2. `fused_recurrent_gated_delta_rule_packed_decode` 加 `gguf_layout: bool = False` 参数
  3. gdn_linear_attn.py `_forward_core_decode_non_spec` 调用处传 `gguf_layout=_GDN_GGUF_LAYOUT`
- 验证:12*8=96 ✓、23*47=1081(竖式)✓、200 字短文 ✓、工具调用(get_weather)✓;
  AWQ 此前也答错('128')证实是共享 div3 路径问题,本修复对 AWQ 同样生效
- 影响:div3 布局(GPTQ8/AWQ)decode 路径;GGUF(mod16)行为不变(% H 保持)

### 2. ~~marlin_utils.py — marlin_permute_scales 支持非 64/32 组(2026-08-10)~~ ⛔ 已回滚
- 结论:误诊。vLLM 内部 `transform_w_s` 先 permute 成 [groups, n] 才调 marlin_permute_scales,
  旧版代码本就正确;判别脚本传错布局 [n, groups] 导致误判。补丁已回滚,文件恢复原状。
  教训:判别脚本必须复刻 vLLM 真实调用链,输入布局假设错误是最高危错误源。

### 1. qwen3_5.py — 纯文本 ForCausalLM 补 IsHybrid(2026-08-10)
- 文件:vllm/model_executor/models/qwen3_5.py
- 修改:
  1. `Qwen3_5ForCausalLM` 继承增加 `IsHybrid`(原 `Qwen3_5ForCausalLMBase` 无此标记)
  2. `Qwen3_5ForCausalLMBase` 新增 `get_mamba_state_dtype_from_config` / `get_mamba_state_shape_from_config`(原只在多模态 `Qwen3_5ForConditionalGeneration` 中,纯文本类缺失导致 `is_hybrid=False` → mamba_block_size 不推导 → `get_kv_cache_spec` 断言崩溃)
- 原因:GGUF 纯文本(Qwen3_5ForCausalLM)也含 linear_attn 混合层,必须标记 IsHybrid
- 兼容性:GGUF(纯文本)修复;AWQ(多模态)走原类不受影响。标注 [FORK 兼容]
- 上游对照:原版 Qwen3_5ForConditionalGeneration 继承 IsHybrid,纯文本类是 fork 加的,属 fork 补漏

## 历史修改(此前会话,登记在案)

(此处按需补录 prior sessions 的 [FORK 兼容] 改动清单)
