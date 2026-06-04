# Qwen3.6 27B KV Throughput Sweep

This document records the FP8 and GPTQ-INT4 checkpoint sweep on the dual RTX
2080 Ti TP=2 runtime. Throughput is reported as `prefill / decode tok/s`.

## Test Matrix

| Checkpoint route | Mode | KV precision | PP4096/TG128 prefill / decode | PP65536/TG512 prefill / decode |
|---|---|---|---:|---:|
| Qwen3.6 27B FP8 | noMTP | FP16 | 1677.0 / 32.9 | 1303.9 / 29.1 |
| Qwen3.6 27B FP8 | noMTP | INT8 | 1663.0 / 33.5 | 1274.8 / 33.7 |
| Qwen3.6 27B FP8 | noMTP | TQK8V4 | 1612.1 / 30.3 | 1277.9 / 20.7 |
| Qwen3.6 27B FP8 | noMTP | TQ4NC | 1656.4 / 29.9 | 1273.9 / 19.6 |
| Qwen3.6 27B FP8 | MTP3 | FP16 | 1614.1 / 76.0 | 1253.9 / 70.8 |
| Qwen3.6 27B FP8 | MTP3 | INT8 | 1609.9 / 74.4 | 1227.9 / 42.8 |
| Qwen3.6 27B FP8 | MTP3 | TQK8V4 | 1614.6 / 70.4 | 1231.7 / 36.9 |
| Qwen3.6 27B FP8 | MTP3 | TQ4NC | 1602.8 / 71.7 | 1227.2 / 33.6 |
| Qwen3.6 27B GPTQ-INT4 | noMTP | FP16 | 1795.5 / 50.3 | 1364.3 / 42.1 |
| Qwen3.6 27B GPTQ-INT4 | noMTP | INT8 | 1781.8 / 51.7 | 1345.8 / 52.2 |
| Qwen3.6 27B GPTQ-INT4 | noMTP | TQK8V4 | 1784.1 / 45.4 | 1358.4 / 26.6 |
| Qwen3.6 27B GPTQ-INT4 | noMTP | TQ4NC | 1778.9 / 43.5 | 1348.4 / 24.9 |
| Qwen3.6 27B GPTQ-INT4 | MTP3 | FP16 | 1732.9 / 81.7 | 1328.0 / 85.5 |
| Qwen3.6 27B GPTQ-INT4 | MTP3 | INT8 | 1716.9 / 95.5 | 1290.7 / 50.1 |
| Qwen3.6 27B GPTQ-INT4 | MTP3 | TQK8V4 | 1726.0 / 94.6 | 1309.1 / 40.2 |
| Qwen3.6 27B GPTQ-INT4 | MTP3 | TQ4NC | 1701.9 / 88.3 | 1294.9 / 36.9 |

## Readout

- With MTP enabled, FP16/default KV keeps the best long-context decode path.
  GPTQ-INT4 MTP3 reaches `1328.0 / 85.5 tok/s` at PP65536/TG512.
- Without MTP, FP16 and INT8 KV do not show a clear long-context decode-speed
  penalty in this sweep. GPTQ-INT4 noMTP FP16 is `1364.3 / 42.1 tok/s`, while
  GPTQ-INT4 noMTP INT8 is `1345.8 / 52.2 tok/s` at PP65536/TG512.
- TurboQuant KV remains fast in short-context MTP tests, but it shows a clear
  long-context decode drop. Use TQK8V4/TQ4NC when the priority is larger KV
  capacity, not maximum long-context decode speed.
- The FP8 MTP3 + INT8 KV PP65536/TG512 row needs a tight request-sized
  `MAX_MODEL_LEN=66048`. A wider `69632` profile OOMs on this 22GB/card setup.

## Artifacts

Raw JSONL artifacts and local CSV summaries are kept outside the source tree.
This file is the public, normalized summary used by the repository docs.
