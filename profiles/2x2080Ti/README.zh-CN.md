# 2x2080Ti Profiles

语言：[English](README.md) | 简体中文

下列 profile 沿用 0.2.x 的配置，待完成 0.3.x Docker 验证。0.3.x Docker 实测见 [2xT10](../2xT10/README.zh-CN.md) 和 [4xT10](../4xT10/README.zh-CN.md)。

## Qwen3.8-27B-FP8

[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)

- [`qwen38-27b/w8a16/mtp4-fp16kv-1x148K-text-only.env`](qwen38-27b/w8a16/mtp4-fp16kv-1x148K-text-only.env)
- [`qwen38-27b/w8a16/nomtp-fp16kv-1x176K-text-only.env`](qwen38-27b/w8a16/nomtp-fp16kv-1x176K-text-only.env)
- [`qwen38-27b/w8a16/nomtp-fp16kv-1x121K-text-image.env`](qwen38-27b/w8a16/nomtp-fp16kv-1x121K-text-image.env)
- [`qwen38-27b/w8a16/mtp4-fp8kv-1x262K-text-only.env`](qwen38-27b/w8a16/mtp4-fp8kv-1x262K-text-only.env)
- [`qwen38-27b/w8a16/mtp4-fp8kv-1x186K-text-image.env`](qwen38-27b/w8a16/mtp4-fp8kv-1x186K-text-image.env)
- [`qwen38-27b/w8a16/yarn-fp8kv-1x338K-text-only.env`](qwen38-27b/w8a16/yarn-fp8kv-1x338K-text-only.env)

## Qwen3.8-27B-NVFP4

[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) · [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)

- [`qwen38-27b/w4a16/dflash2-fp8kv-1x262K-text-image.env`](qwen38-27b/w4a16/dflash2-fp8kv-1x262K-text-image.env)
- [`qwen38-27b/w4a16/mtp4-fp8kv-2x229K-text-only.env`](qwen38-27b/w4a16/mtp4-fp8kv-2x229K-text-only.env)
- [`qwen38-27b/w4a16/yarn-fp8kv-1x524K-text-only.env`](qwen38-27b/w4a16/yarn-fp8kv-1x524K-text-only.env)
- [`qwen38-27b/w4a16/dflash-fp8kv-2x176K-text-only.env`](qwen38-27b/w4a16/dflash-fp8kv-2x176K-text-only.env)
- [`qwen38-27b/w4a16/mtp4-tq4nc-3x262K-text-only.env`](qwen38-27b/w4a16/mtp4-tq4nc-3x262K-text-only.env)

## Qwen3.6-35B-A3B-FP8

[Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

- [`qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env`](qwen35b/w8a16/nomtp-fp16kv-1x262K-text-only.env)
- [`qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env`](qwen35b/w8a16/nomtp-fp8kv-1x221K-text-image.env)
