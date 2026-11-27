ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
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

COPY tools/flashqla_sm75_patches/ /tmp/flashqla-sm75/
COPY docker/sm75/flashqla-LICENSE /usr/share/doc/vllm-sm75/flashqla-LICENSE
COPY LICENSE /usr/share/doc/vllm-sm75/LICENSE

# Build against the exact Torch and CUDA shipped in the upstream image.
RUN set -eux; \
    site=$(python3 -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file(""))'); \
    python3 -c 'import importlib.metadata as m, sys; assert m.version("vllm").split("+")[0] == sys.argv[1]' "$UPSTREAM_VERSION"; \
    TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS="$MAX_JOBS" python3 -c 'from torch.utils.cpp_extension import load; load(name="flash_qla_legacy_gdn", sources=["/tmp/flashqla-sm75/gdn_forward.cu"], build_directory="/tmp/flashqla-sm75", extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])'; \
    install -m 0644 /tmp/flashqla-sm75/flash_qla_legacy_gdn.so "$site/flash_qla_legacy_gdn.so"; \
    install -m 0644 /tmp/flashqla-sm75/sm_legacy.py "$site/vllm_sm75_flashqla.py"; \
    python3 -c 'from vllm_sm75_flashqla import _validate_extension; import flash_qla_legacy_gdn; _validate_extension(flash_qla_legacy_gdn)'; \
    rm -rf /tmp/flashqla-sm75

RUN --mount=type=bind,source=docker/sm75/patches,target=/vllm-sm75-patches \
    set -eux; \
    site=$(python3 -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file(""))'); \
    for file in /vllm-sm75-patches/*.patch; do \
        patch --directory="$site" --strip=1 --batch --forward --fuzz=0 < "$file"; \
    done

FROM scratch AS inkling-sources
ADD --checksum=sha256:0a9c5e743c075c728dea20c2d3fca1b23ac7da09324bcc4e74302c1a036ec6f1 \
    https://codeload.github.com/lvb-s/vllm-inkling-sm75/tar.gz/811e9b7f40fd9c6990a6b7ed2a1016ef050f0c36 /community.tar.gz
ADD --checksum=sha256:a80232fce11e29c643f8e26604bb8740db40861ef193fb1ea2e94bb57a79ca99 \
    https://codeload.github.com/vllm-project/vllm/tar.gz/a289c722dd59a0e74bf77e005a7077676c320f5c /upstream.tar.gz
ADD --checksum=sha256:d4747cae223f81bb8d18bee766a5f41e67484a65bd665534fdbc8dc7adabd7c1 \
    https://files.pythonhosted.org/packages/53/b5/19eaeed76d26248b8fe9276d2f9a4346e250fd7d2cfb0784ae4929d7efcc/humming_kernels-0.1.20-cp310-abi3-manylinux_2_28_x86_64.whl /

FROM runtime
LABEL io.github.weicj.vllm.inkling-source="https://github.com/vllm-project/vllm/tree/a289c722dd59a0e74bf77e005a7077676c320f5c" \
      io.github.weicj.vllm.inkling-sm75-source="https://github.com/lvb-s/vllm-inkling-sm75/tree/811e9b7f40fd9c6990a6b7ed2a1016ef050f0c36"

RUN --mount=type=bind,from=inkling-sources,source=/,target=/inkling-sources \
    --mount=type=bind,source=docker/sm75/inkling,target=/inkling-patches \
    set -eux; \
    site=$(python3 -c 'import importlib.metadata as m; print(m.distribution("vllm").locate_file(""))'); \
    authors=/usr/share/doc/vllm-sm75; \
    community=vllm-inkling-sm75-811e9b7f40fd9c6990a6b7ed2a1016ef050f0c36; \
    upstream=vllm-a289c722dd59a0e74bf77e005a7077676c320f5c; \
    tar -xf /inkling-sources/community.tar.gz -C "$site" --strip-components=2 \
        "$community/overlay/vllm/models/inkling2"; \
    tar -xf /inkling-sources/upstream.tar.gz -C "$site" --strip-components=1 \
        "$upstream/vllm/models/inkling2/nvidia/ops/trtllm_bmm.cu" \
        "$upstream/vllm/models/inkling2/speculator_config.py"; \
    tar -xOf /inkling-sources/community.tar.gz "$community/LICENSE" > "$authors/inkling-sm75-LICENSE"; \
    tar -xOf /inkling-sources/upstream.tar.gz "$upstream/LICENSE" > "$authors/inkling-upstream-LICENSE"; \
    patch --directory="$site" --strip=1 --batch --forward --fuzz=0 < /inkling-patches/inkling-v0.34.0.patch; \
    python3 -m pip install --no-deps --no-index --no-cache-dir /inkling-sources/humming_kernels-*.whl; \
    python3 -m compileall -q "$site/vllm/models/inkling2"
