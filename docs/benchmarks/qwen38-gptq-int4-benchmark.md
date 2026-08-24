# Qwen3.8-27B GPTQ INT4 Benchmark Results

**Date**: 2026-08-21

**Hardware**: 2x RTX 2080 Ti 22GB, NVLink (NV2)

**Software**: vLLM v0.1.16 (weicj/vLLM-2080Ti-Definitive), base vLLM 0.21.0

**Model**: [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)

**Model Size**: 18.22 GB (5 safetensors shards)

**Quantization**: GPTQ INT4, group_size=128, sym=True, desc_act=False

**MTP**: Preserved in BF16

## Benchmark Results (PP4096/TG128)

### Configuration A: FP16 KV + MTP3 (Speed Priority)

| Metric | Value |
|--------|-------|
| Prefill | 1690.7 tok/s |
| **Decode** | **126.2 tok/s** |
| Max Context | ~128K (estimated) |
| KV Cache | FP16 |
| MTP K | 3 |
| Idle VRAM/GPU | ~20 GB |

### Configuration B: INT8 KV + noMTP (Context Priority)

| Metric | Value |
|--------|-------|
| Prefill | 1741.4 tok/s |
| Decode | 54.7 tok/s |
| **Max Context** | **128K (confirmed)** |
| KV Cache | INT8 |
| MTP K | 0 |
| Idle VRAM/GPU | ~21.3 GB |

## Comparison with Qwen3.6-27B GPTQ INT4

| Config | Qwen3.6 4K PP/TG | Qwen3.8 4K PP/TG | Delta |
|--------|-----------------|-----------------|-------|
| FP16 KV + noMTP | 1795.5 / 50.3 | N/A | - |
| INT8 KV + noMTP | 1781.8 / 51.7 | 1741.4 / 54.7 | -2.3% / +5.8% |
| FP16 KV + MTP3 | 1732.9 / 81.7 | **1690.7 / 126.2** | -2.4% / **+54.5%** |

**Key Finding**: Qwen3.8 GPTQ INT4 with FP16 KV + MTP3 reaches **126.2 tok/s decode** at 4K PP/TG, **+54.5% vs Qwen3.6** (81.7 tok/s) measured with the same 4K PP/TG benchmark口径.

## GPTQ Marlin Verification

- Marlin kernel: ✅ SM75 supported
- GPTQMarlinConfig: ✅ Correctly detected
- Qwen3_5ForConditionalGeneration: ✅ Architecture resolved
- MTP head: ✅ Detected and loaded
- TP=2: ✅ Working

## Recommendations

| Use Case | Config | Reason |
|----------|--------|--------|
| **Codex / OpenCode Agent** | Config A | 2.3x faster decode (126.2 vs 54.7 tok/s) |
| **Long Context** | Config B | 128K context confirmed, likely 256K+ with YaRN |
| **General Use** | Config A | Best decode speed with adequate context |

## Startup Command (Config A)

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 8000 \
  --model /path/to/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --served-model-name qwen38-gptq-fp16kv-128K-mtp3-text-only \
  --dtype half \
  --tensor-parallel-size 2 \
  --generation-config vllm \
  --max-model-len 128000 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --skip-mm-profiling \
  --disable-log-stats \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}'
```

## Notes

1. GPTQ quantization is auto-detected from model directory name containing "GPTQ"
2. Marlin kernel is used for INT4 GEMM operations on SM75
3. MTP head is preserved in BF16 for speculative decoding
4. The model supports both text-only and text+image modes
5. Only the 128K profiles are shipped and validated for Qwen3.8-27B in this PR. Long-context (>128K) INT8 KV / YaRN profiles exist for qwen27b, not yet for qwen38
