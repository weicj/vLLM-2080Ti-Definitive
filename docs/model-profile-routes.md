# Model Profile Routes

This document records the deployment profile choices for the dual RTX 2080 Ti
SM75 runtime. Limits are listed before the profile matrices because they define
where each route is safe to use.

Throughput details are kept in
[Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
[MTP Task Sensitivity](mtp-task-sensitivity.md).

## Qwen3.6 27B

### Limits

- INT8 KV is a text-serving capacity route. It is not recommended for image
  serving because validated multimodal runs reached READY but produced
  corrupted punctuation/output instead of stable image answers.
- FP16/default KV has a real `PP262000/TG1` pass only in noMTP mode. The MTP3
  262K service can start, but the real 262K prompt OOMs during prefill, so MTP3
  stays a short-context speed route for FP16.
- Multi-workspace profiles are for queued workspace isolation, not true parallel
  long-prefill throughput. This TP=2 runtime still serializes heavy long-context
  work in practice.
- YaRN 524K is an offline capacity profile. Native 262K profiles remain the
  default for normal interactive serving.
- FP8 is a weight route, not a KV-precision route. On SM75 it uses weight-only
  FP8 rather than native FP8 tensor-core compute, so it is the highest-quality
  practical 8-bit Qwen route while AWQ/GPTQ-INT4 remains the default
  performance/capacity family.
- Larger MTP values can produce higher synthetic throughput-only numbers. MTP3
  is the practical deployment reference because it is better balanced across
  acceptance rate and real workloads.

### Profiles

| Use case | Weight quantization | KV precision | Context | Spec decoding | Message type | Concurrency limit |
|---|---|---|---|---|---|---|
| Highest-quality 8-bit text route | FP8 | FP16 | 8K-64K validated | Native MTP3 | text | 1 request |
| High-quality native-context route | AWQ/GPTQ-INT4 | FP16 | 262K native | None | text | 1 request |
| Peak short-context speed route | AWQ/GPTQ-INT4 | FP16 | 8K-16K | Native MTP3 | text | 1 request |
| High-compression route | AWQ/GPTQ-INT4 | TQ4NC | 262K native | Native MTP3 | text | 1 request / queued |
| Ultra-long context | AWQ/GPTQ-INT4 | INT8 | 524K YaRN | Native MTP3 | text | 1 offline request |
| Multi-workspace | AWQ/GPTQ-INT4 | INT8 or TQ4NC | 64K-262K caps | Native MTP3 | text | 4 x 64K queued / 2 x 262K queued |
| Multimodal | AWQ/GPTQ-INT4 | TQ4NC | 262K native | Native MTP3 | text + image | 1 request |

### Launcher Presets

- `qwen27-awq-mtp3`: regular Qwen-family FP16/default KV + native MTP3 route.
- `qwen27-gptq-mtp3`: GPTQ-INT4 Qwen-family FP16/default KV + native MTP3 route.
- `qwen27-fp8-mtp3`: FP8 Qwen-family FP16/default KV + native MTP3 route.
- `qwen27-awq-mtp3-peak`: short-context peak-speed text route.
- `qwen27-awq-mtp3-tq4nc-fi`: TQ4NC compressed-capacity route.
- `qwen27-awq-mtp3-int8kv`: INT8 KV capacity / YaRN route.
- `qwen27-awq-mm-*` and `heretic27-gptq-mm-*`: image-serving experiment
  presets for Qwen-family model variants.

## Gemma4 31B

### Limits

- Gemma4 uses head_dim=512 with heterogeneous/GQA attention. On SM75, the
  compressed-KV path does not have a validated FlashAttention/FlashInfer fast
  prefill route; the 262K INT8 path currently falls back to SDPA/GQA and is
  much slower than the FP16/default-KV route.
- The practical Gemma context window is much smaller than Qwen on the same dual
  22GB hardware. The combination of head_dim=512, GQA/heterogeneous attention,
  and less efficient compressed-KV grouping means that starting a compressed-KV
  profile is not the same as getting an efficient full-262K service profile.
- FP16/default KV is the best speed and quality route, but it cannot reach the
  full native 262K context on dual 22GB cards. The validated service profile is
  16K; the `105216` text and `97152` image figures are startup estimates from
  failed 262K probes, not proven practical request limits.
- TQ4NC is kept as a fast compressed short-context route. The useful real-prompt
  edge is about `43K`: a `43005`-token prompt passed, while `43505`/`44005`
  failed admission. `64K` is READY-only evidence, not a proven long-prompt pass.
- Gemma multimodal is kept on default KV. INT8/TQ4NC multimodal routes are not
  recommended because the heterogeneous-head multimodal backend rejects or
  breaks those compressed KV paths.

### Profiles

| Use case | Weight quantization | KV precision | Context | Spec decoding | Message type | Concurrency limit |
|---|---|---|---|---|---|---|
| High-quality route | GPTQ-INT4 | FP16 | 16K validated text service; 105K estimate only | Assistant MTP5 | text | 1 request |
| Fast compressed route | GPTQ-INT4 | TQ4NC | 43K real-prompt practical edge | None | text | 1 request |
| Long-context route | GPTQ-INT4 | INT8 | 262K native, slow offline | None | text | 1 slow offline request |
| Multimodal compatibility | GPTQ-INT4 | FP16 | 8K image validated | Assistant MTP3 compatible | text + image | 1 request |

### Launcher Presets

- `gemma4-gptq-tq4nc-mtp3`: TQ4NC compatibility route with assistant MTP3.
- `gemma4-gptq-tq4nc-nomtp`: TQ4NC no-MTP short-context route.
- `gemma4-gptq-mm-nomtp`: default-KV image-serving compatibility route.
- `gemma4-gptq-mm-mtp3`: default-KV image-serving route with assistant MTP3.
