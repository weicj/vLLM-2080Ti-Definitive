# EXL3 on SM75

`vLLM-2080Ti-Definitive` contains the loader support for EXL3 PLE checkpoints.
It does not package an EXL3 CUDA runtime or a quantization plugin. Install the
two pinned external components into this checkout's existing virtual
environment:

```bash
git clone --branch turing-sm75 https://github.com/weicj/exllamav3-turing.git
git clone --branch turing-runtime https://github.com/weicj/vllm-exl3-turing.git
export EXL3_NGRAM_STREAM=1
./vllm-exl3-turing/scripts/install_into_vllm_2080ti.sh \
  "$PWD" ./exllamav3-turing
```

The runtime revision is pinned by
`vllm-exl3-turing/runtime/EXLLAMAV3_TURING.lock`. Keep that lock and the
installed runtime together. Do not install a generic `exllamav3` or
`vllm-exl3` wheel into this environment: either can replace the SM75 extension
or resolve an upstream vLLM distribution over this fork.

`EXL3_NGRAM_STREAM=1` is required for Qwen Flash PLE packs. It keeps the
n-gram embedding table in the runtime's SSD streaming mode; non-filtering
checkpoint loaders are deliberately rejected by the vLLM loader integration.
