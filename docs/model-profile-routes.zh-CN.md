# 模型 Profile 路线

本文定义 0.2.x 部署 profile 的证据口径。当前路线清单维护在 [profiles/README.zh-CN.md](../profiles/README.zh-CN.md)，迁移证据记录在 [0.2.x 验证报告](2080ti-0.2.1-pre-validation.md)中。

`Profile` 是 `launcher.sh` 选择的相对 `.env` 路线；checkpoint、显卡选择、端口、chat template 和 reasoning 默认值仍由 launcher 管理。

## 证据口径

正式推荐的路线必须完成真实请求，返回 HTTP 200、正常结束 stream，并通过文档规定的质量 smoke。只 load 成功、health/READY 通过、空 stream 或极小窗口 smoke 都不能作为容量或吞吐证据。

每条 benchmark 记录必须注明确切 checkpoint、权重精度、KV 精度、MTP 设置、上下文长度、graph mode、TP/PP 布局和测量方法。历史 0.1.x CUDA 12.8 / PyTorch 2.11 数据只能作为兼容性证据，不能直接作为 0.2.x CUDA 13.0 的晋升证据。

## 路线策略

- `normal` 是完成对应 0.2.x 验证后使用的默认生产路线。
- `fast` 只用于通过质量 smoke 和非 eager CUDA Graph 验证的路线。
- `safe` 是诊断或兼容性回退档，不是性能推荐路线。
- FP16/default KV 是质量参考；INT8 和 TurboQuant KV 路线必须分别完成质量和容量验证。
- 实验性或未验证路线必须明确标注，不能仅凭 profile 名称晋升为推荐配置。

对 [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)，
实验性 TP2xPP4 路线有一组 6 张 T10 + 2 张 RTX 2080 Ti 的异构启动/容量参考：
`GPU_UTIL=0.92` 下为 152,492 GPU KV tokens。该记录使用 NVFP4 权重、FP16 KV、MTP=0、
100K 上下文和非 eager CUDA Graph，测量方式是启动/健康检查/真实请求容量探测，不是吞吐测试。
