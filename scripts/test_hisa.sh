#!/usr/bin/env bash

set -euo pipefail

src_dir=${LLAMA_CPP_SOURCE_DIR:-/build/llama.cpp}
build_dir=${LLAMA_CPP_BUILD_DIR:-/building}
skip_pull=${LLAMA_CPP_SKIP_PULL:-0}
ctest_regex=${LLAMA_CPP_CTEST_REGEX:-test-backend-ops|test-quantize-fns|test-hisa-graph-smoke}

if [[ ! -d "${src_dir}" ]]; then
    echo "error: source directory not found: ${src_dir}" >&2
    exit 1
fi

if [[ ! -d "${build_dir}" ]]; then
    mkdir -p "${build_dir}"
fi

if [[ "${skip_pull}" != "1" ]]; then
    git -C "${src_dir}" pull --ff-only
fi

LLAMA_CPP_SOURCE_DIR="${src_dir}" \
LLAMA_CPP_BUILD_DIR="${build_dir}" \
LLAMA_CPP_CTEST_REGEX="${ctest_regex}" \
bash "${src_dir}/scripts/cuda-ctest.sh"