## Qwen3.8 Flash-Next

Model card: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)

This PR is a functional validation of the Qwen4-Exp + NVFP4/Marlin + PLE
path on SM75. It is experimental and is not a production-performance claim.
The throughput below uses the repository's fixed 4K-input/128-output shape for
comparison; real workloads can fall to single-digit tok/s, especially with
complex multi-GPU topology and CPU limits.

| Profile | Context / GPU util | GPU KV tokens | 4K/128 median (prefill / decode) | Recommendation |
|---|---:|---:|---:|---|
| `qwen38flashnext/w4a16/experimental/tp2pp4-fp16kv-nomtp-text.env` | 100K / 0.92 | 102,591 | 2379.14 / 24.92 tok/s | Recommended balance |
| `qwen38flashnext/w4a16/experimental/tp4pp2-fp16kv-nomtp-text.env` | 90K / 0.96 | 100,031 | 1697.99 / 28.18 tok/s | Decode priority |
| TP2xPP4 heterogeneous reference | 100K / 0.92 | 152,492 | — | 6× T10 + 2× RTX 2080 Ti; capacity/startup only |

Notes:

- `CUSTOM_ALL_REDUCE_MODE=auto` enables CAR only for fully NVLink-connected,
  P2P-valid TP groups. PCIe-only groups remain on NCCL/PYNCCL.
- Usage: select eight GPUs, choose the TP/PP profile, and confirm rank order in
  `launcher.sh`. GPU models may differ between PP stages, but every TP group
  must be internally P2P-valid; check the result with `--print-config`.
