ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG UPSTREAM_VERSION
ARG MAX_JOBS=2
ARG FORK_RELEASE

LABEL org.opencontainers.image.title="vLLM 2080 Ti Definitive Edition" \
      org.opencontainers.image.authors="github.com/weicj" \
      org.opencontainers.image.source="https://github.com/weicj/vLLM-2080Ti-Definitive" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.version="${FORK_RELEASE}" \
      io.github.weicj.vllm.upstream-version="${UPSTREAM_VERSION}"

COPY docker/sm75/patches/ /tmp/vllm-sm75-patches/
COPY tools/flashqla_sm75_patches/ /tmp/flashqla-sm75/
COPY docker/sm75/flashqla-LICENSE /usr/share/doc/vllm-sm75/flashqla-LICENSE
COPY LICENSE /usr/share/doc/vllm-sm75/LICENSE

# Build against the exact Torch and CUDA shipped in the upstream image.
RUN set -eux; \
    site=$(python3 -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file(""))'); \
    python3 -c 'import importlib.metadata as m, sys; assert m.version("vllm").split("+")[0] == sys.argv[1]' "$UPSTREAM_VERSION"; \
    for file in /tmp/vllm-sm75-patches/*.patch; do \
        patch --directory="$site" --strip=1 --batch --forward --fuzz=0 < "$file"; \
    done; \
    TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS="$MAX_JOBS" python3 -c 'from torch.utils.cpp_extension import load; load(name="flash_qla_legacy_gdn", sources=["/tmp/flashqla-sm75/gdn_forward.cu"], build_directory="/tmp/flashqla-sm75", extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])'; \
    install -m 0644 /tmp/flashqla-sm75/flash_qla_legacy_gdn.so "$site/flash_qla_legacy_gdn.so"; \
    install -m 0644 /tmp/flashqla-sm75/sm_legacy.py "$site/vllm_sm75_flashqla.py"; \
    python3 -c 'from vllm_sm75_flashqla import _validate_extension; import flash_qla_legacy_gdn; _validate_extension(flash_qla_legacy_gdn)'; \
    rm -rf /tmp/vllm-sm75-patches /tmp/flashqla-sm75
