# Model Profile Routes

This document records the evidence rules behind the deployment profiles. The
current profile catalog, meanings, and measured throughput live in
[Profile Guide](../profiles/README.md).

`Profile` is the relative `.env` path selected by `launcher.sh`; the concrete
checkpoint is selected separately with `MODEL_DIR`.

## Evidence Rule

Full pass means a real request returns HTTP 200, finishes the stream, and the
Chinese quality smoke has no repetition collapse, broken output, garbling, or
clear wrong-response behavior.

A memory plateau only proves lower capacity risk. If quality smoke fails, the
route is not promoted as a profile even when capacity or synthetic throughput
looks usable.

Load-only, READY-only, health-only, small-window smoke, and empty-stream results
are not capacity evidence.

## KV Positioning

- FP16/default KV is the quality route.
- INT8 KV is the capacity / balance route; currently shipped only as
  `normal` / piecewise profiles.
- TQK8V4 is the TurboQuant compression route. It is shipped for quality-passed
  `fast` profiles and for the Qwen3.8 NVFP4 DFlash2 `normal` route, whose
  speculative target verification uses the causal safe path.
- TQ4NC had capacity experiments, but is not used in the current shipped
  profiles.

## Notes

- Profiles are organized as `profiles/<model>/<mode>/<weight>/<route>.env`.
- `normal` is the current recommended production route. `fast` keeps only
  high-performance routes that passed quality smoke. `safe` is the launcher
  eager fallback mode, not the current shipped profile directory.
- The same dual-2080-Ti runtime also validates Qwen3.6 35B FP8 MoE lanes. The
  shipped preset set now covers 256K `normal` and `aggressive` noMTP
  text-only lanes, 136K `normal` and `aggressive` noMTP text+image lanes, and
  a 178K `fast` MTP3 speed preset.
- FP8 + FP16KV `normal` is formally 256K and passes a long-prompt smoke at
  `262016/128`.
- FP8 + FP16KV `aggressive` is also validated at 256K. Its recorded throughput
  stays on the `4096/128` synthetic lane, while the near-full `262016/128`
  smoke is treated as a capacity proof because stream chunk coalescing can
  overstate long-run decode speed.
- FP8 + FP16KV text+image is validated at 136K in both `normal` and
  `aggressive`. Both passed `138240/128`; `139008/64` also passed at the edge,
  while `139136/32` exceeds the configured `139264` limit.
- FP8 + FP16KV `fast` is currently validated at 178K and passes a long-prompt
  smoke at `182144/128`.
- FP8 + TQK8V4 is validated at 256K for text-only and 240K for text-image.
  The image route uses GPU util 0.96.
- Qwen3.8 27B NVFP4 + DFlash2 is the first promoted DFlash2 route. It uses
  `nvidia/Qwen3.8-27B-NVFP4`, `incoai/Qwen3.8-27B-DFlash2`, TQK8V4, K=7,
  TRITON_ATTN for the draft, and PIECEWISE CUDA Graphs. The SM75 target
  verification path preserves causal sequence lengths and uses B=8 chunks.
  Three independent high-acceptance fixed 4K/128 runs measured 155.16,
  166.57, and 166.76 decode tok/s (mean 162.83). The route passed 4K
  correctness requests, a near-full 262016-token prompt smoke, and real 4K
  and 8K-token HTML/JavaScript generations without a stream failure, CUDA
  illegal instruction, or engine death. It is text-only and formally 256K.
- fast + INT8KV is not kept: capacity or synthetic throughput may pass, but
  Chinese quality smoke showed repeated or broken output.
- Qwen3.8 Flash-Next NVFP4 is identified as the `qwen4` architecture and kept
  as an experimental eight-T10 PP family. Its shipped profiles use validated
  `PLE_PLACEMENT=disk`; `cpu` and `gpu` remain explicit opt-in placements. In
  the current environment, the recommended TP2xPP4 profile at
  `GPU_UTIL=0.92` measured 102,591 GPU KV tokens. After QSA JIT warmup, ten
  sequential fixed 4K/128 runs measured a 2379.14 / 24.92 tok/s median and a
  2469.26 / 24.90 tok/s mean. TP4xPP2 at `GPU_UTIL=0.96` measured 100,031 GPU
  KV tokens and a 1697.99 / 28.18 tok/s median (1652.15 / 27.33 mean). Eight
  of ten decode samples clustered around 28.3 tok/s. The historical 33.93
  tok/s TP4xPP2 sample is shape-dependent and is not a stable promise.
- The profile's CAR `auto` mode opts into topology detection. CAR is selected
  only when every GPU in the TP group is both P2P-reachable and fully connected
  by NVLink (or XGMI on ROCm). PCIe-only T10
  groups remain on NCCL/PYNCCL, which is the validated decode path.
- Current-environment capacity evidence is 102,591 GPU KV tokens at
  `GPU_UTIL=0.92` for homogeneous eight-T10 TP2xPP4 and 100,031 at
  `GPU_UTIL=0.96` for homogeneous eight-T10 TP4xPP2. A heterogeneous TP2xPP4
  startup reference (six T10 + two RTX 2080 Ti; TP groups `[0,2]`, `[8,9]`,
  `[6,7]`, `[1,5]`) reached **152,492** tokens at `GPU_UTIL=0.92` and passed
  health/real requests. Its launcher log is pre2-labelled, so re-run on the
  current pre3 tree before treating it as a pre3 capacity promotion.
  TP2xPP4 at `0.95` exposed 115,834 tokens, but its ten-run 4K/128 median
  degraded to 1274.87 / 21.15 tok/s. At `0.97`, it reached 124,662 startup
  tokens but failed the real request in a FlashQLA temporary allocation.
- Legacy `fast` + INT8KV compatibility issues should be reported and fixed, but
  those fixes do not promote the route back into the shipped fast catalog.
- Throughput background is kept in
  [Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md) and
  [MTP Task Sensitivity](mtp-task-sensitivity.md).
