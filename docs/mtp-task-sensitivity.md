# MTP Task Sensitivity

MTP/speculative decoding is not a fixed multiplier. It speeds up decoding only
when the draft tokens are accepted by the target model. More speculative tokens
can improve synthetic decode throughput, but they can also waste work when the
task causes low acceptance.

The stable deployment rule for this fork is therefore:

- Use conservative MTP settings for mixed agent workloads.
- Treat higher MTP values as profile-specific speed options.
- Validate MTP with real prompts, not only repeated-token synthetic tests.

## Why Acceptance Matters

Speculative decoding drafts several future tokens, then the target model accepts
or rejects them. When many draft tokens are accepted, one target-model step can
produce multiple output tokens. When acceptance drops, the draft model still
consumes compute, but the target model accepts fewer tokens, so decode speed can
stagnate or regress.

This is why MTP performance depends on:

- task type and output entropy,
- model and draft quality,
- number of speculative tokens,
- KV dtype and graph/fallback path,
- whether the benchmark excludes cold-start/JIT overhead.

## LongGen3 4096/1024 Sweep

This sweep uses three Chinese long-generation prompts with about 4096 prompt
tokens and up to 1024 generated tokens each. Cold start, model load, graph
capture, and first-request JIT are excluded.

### Qwen3.6-27B-GPTQ-INT4

This GPTQ-INT4 Qwen-family route has the stronger noMTP baseline in the same
LongGen3 TTFT test. It still shows clear MTP scaling, but the useful plateau
starts around MTP3/MTP4.

| MTP | Decode tok/s | Relative to noMTP |
|---|---:|---:|
| noMTP | 43.59 | 1.00x |
| MTP1 | 42.94 | 0.99x |
| MTP2 | 49.11 | 1.13x |
| MTP3 | 60.62 | 1.39x |
| MTP4 | 60.77 | 1.39x |
| MTP5 | 59.55 | 1.37x |

Per-case behavior at the top end is close: MTP4 reached `63.10 tok/s` on code
and `64.57 tok/s` on science, while MTP3 was faster on prose at `58.14 tok/s`.
This is why MTP3 remains the safer mixed-workload default.

### Gemma4-31B-GPTQ-INT4

Gemma also benefits from MTP, but the gain is smaller and more workload
sensitive. MTP7 was numerically best in this sweep, while MTP8 was the first
decline.

| MTP | Decode tok/s | Relative to noMTP |
|---|---:|---:|
| noMTP | 31.65 | 1.00x |
| MTP1 | 21.47 | 0.68x |
| MTP2 | 28.92 | 0.91x |
| MTP3 | 33.71 | 1.07x |
| MTP4 | 36.64 | 1.16x |
| MTP5 | 39.06 | 1.23x |
| MTP6 | 40.26 | 1.27x |
| MTP7 | 41.19 | 1.30x |
| MTP8 | 39.11 | 1.24x |
| MTP9 | 38.12 | 1.20x |
| MTP10 | 38.56 | 1.22x |

The overall gain is real, but it is far from a universal 2x decode multiplier.
Gemma high-K MTP should be treated as a tuned profile, not a default rule.

## LongGen3 Per-Task Split

This is the complete LongGen3 4096/1024 TTFT-split sweep. Each row excludes
cold start, model load, graph capture, and first-request JIT. `Prefill TTFT` is
prompt tokens divided by TTFT; `Decode` is completion tokens divided by decode
time after TTFT.

The three task columns are per-case decode tok/s:

- `Code`: simple Python program generation.
- `Science`: scientific-computing analysis.
- `Prose`: Chinese romance/natural prose generation.

### Qwen3.6-27B-GPTQ-INT4

| MTP | Prefill TTFT tok/s | Decode tok/s | Code | Science | Prose |
|---|---:|---:|---:|---:|---:|
| noMTP | 1788.36 | 43.59 | 43.27 | 44.11 | 43.27 |
| MTP1 | 1740.58 | 42.94 | 44.05 | 46.35 | 38.98 |
| MTP2 | 1735.05 | 49.11 | 48.01 | 55.19 | 45.04 |
| MTP3 | 1729.25 | 60.62 | 59.95 | 63.87 | 58.14 |
| MTP4 | 1738.42 | 60.77 | 63.10 | 64.57 | 55.31 |
| MTP5 | 1732.00 | 59.55 | 61.16 | 66.67 | 52.41 |

This GPTQ-INT4 Qwen-family route keeps noMTP decode above `40 tok/s` in this
test. MTP3 and MTP4 are effectively tied on aggregate decode, but MTP3 has the
better prose row and is therefore the conservative mixed-workload default.

### Gemma4-31B-GPTQ-INT4

| MTP | Prefill TTFT tok/s | Decode tok/s | Code | Science | Prose |
|---|---:|---:|---:|---:|---:|
| noMTP | 1626.98 | 31.65 | 31.55 | 31.77 | 31.54 |
| MTP1 | 1635.45 | 21.47 | 22.89 | 21.11 | 20.42 |
| MTP2 | 1637.95 | 28.92 | 32.99 | 28.35 | 26.07 |
| MTP3 | 1635.95 | 33.71 | 41.70 | 33.36 | 28.20 |
| MTP4 | 1609.10 | 36.64 | 46.97 | 35.94 | 29.99 |
| MTP5 | 1634.69 | 39.06 | 52.79 | 39.57 | 29.68 |
| MTP6 | 1640.97 | 40.26 | 56.87 | 41.25 | 30.00 |
| MTP7 | 1629.21 | 41.19 | 55.11 | 42.53 | 31.88 |
| MTP8 | 1625.00 | 39.11 | 57.37 | 39.39 | 28.82 |
| MTP9 | 1624.51 | 38.12 | 55.59 | 39.17 | 27.64 |
| MTP10 | 1616.40 | 38.56 | 56.99 | 38.81 | 28.15 |

Gemma benefits more gradually. Code and science keep improving into higher K,
while prose remains weak and tops out around MTP7 in this sweep. The practical
lesson is that Gemma MTP should be selected by workload: higher K can help
structured output, but it is not a universal natural-language speedup.

## Acceptance Evidence

The benchmark JSON records request timing and token counts, but vLLM reports
speculative acceptance in server-side `SpecDecoding metrics` windows. The table
below maps those log windows back to the sequential LongGen3 requests. Treat
these values as log-window evidence for the scored run, not as a separate field
emitted by the benchmark runner.

### Qwen3.6-27B-GPTQ-INT4

| MTP | Accepted | Drafted | Weighted accept | Decode tok/s |
|---|---:|---:|---:|---:|
| MTP1 | 1525 | 1752 | 87.0% | 42.94 |
| MTP2 | 2061 | 2668 | 77.2% | 49.11 |
| MTP3 | 2317 | 3411 | 67.9% | 60.62 |
| MTP4 | 2430 | 4212 | 57.7% | 60.77 |
| MTP5 | 2468 | 4810 | 51.3% | 59.55 |

### Gemma4-31B-GPTQ-INT4

| MTP | Accepted | Drafted | Weighted accept | Decode tok/s |
|---|---:|---:|---:|---:|
| MTP1 | 1394 | 1715 | 81.3% | 21.47 |
| MTP2 | 1872 | 2620 | 71.5% | 28.92 |
| MTP3 | 2034 | 3201 | 63.5% | 33.71 |
| MTP4 | 2100 | 3736 | 56.2% | 36.64 |
| MTP5 | 2229 | 4580 | 48.7% | 39.06 |
| MTP6 | 2265 | 5226 | 43.3% | 40.26 |
| MTP7 | 2424 | 6139 | 39.5% | 41.19 |
| MTP8 | 2350 | 6768 | 34.7% | 39.11 |
| MTP9 | 2386 | 7641 | 31.2% | 38.12 |
| MTP10 | 2259 | 7550 | 29.9% | 38.56 |

This is the main reason the speed table should not be read as "larger K is
always better." Both models show falling acceptance as K increases. Qwen reaches
its practical plateau around MTP3/MTP4, while Gemma keeps gaining until MTP7
because the larger draft depth still offsets the lower acceptance on this
LongGen3 mix.

## Deployment Guidance

Use these settings as practical starting points:

| Route | Recommended MTP | Reason |
|---|---:|---|
| Qwen3.6 mixed agent workloads | MTP3 | Strong gain with less acceptance risk |
| Qwen3.6 code-heavy or synthetic speed checks | MTP4-MTP5 | Slightly higher peak, profile-specific |
| Qwen3.6 very high K such as MTP8 | Avoid by default | Can win synthetic tests but regress real long output |
| Gemma4 mixed workloads | MTP3-MTP5 | MTP5 is faster in tested long output; MTP3 is safer |
| Gemma4 code/science-heavy output | MTP5-MTP6 | Best observed task-class speed |
| Gemma4 natural prose-heavy output | Lower K preferred | Higher K can lose acceptance and flatten speed |

The benchmark result to trust depends on the serving target:

- Use PP4096/TG128 for clean single-request peak evidence.
- Use LongGen3 4096/1024 for balanced prefill+decode behavior.
- Use real agent/router tasks for service-like latency and quality.
- Always exclude cold start, model load, CUDA graph capture, Triton/AOT compile,
  and first-request JIT when comparing MTP settings.
