#!/usr/bin/env bash

set -euo pipefail

src_dir=${LLAMA_CPP_SOURCE_DIR:-/build/llama.cpp}
build_dir=${LLAMA_CPP_BUILD_DIR:-/building}
build_type=${LLAMA_CPP_BUILD_TYPE:-RelWithDebInfo}
build_jobs=${LLAMA_CPP_BUILD_JOBS:-$(nproc)}
ctest_regex=${LLAMA_CPP_CTEST_REGEX:-test-backend-ops|test-quantize-fns}
python_target_dir=${LLAMA_CPP_PYTHON_TARGET_DIR:-}

build_examples=${LLAMA_CPP_BUILD_EXAMPLES:-OFF}
build_tools=${LLAMA_CPP_BUILD_TOOLS:-ON}
build_server=${LLAMA_CPP_BUILD_SERVER:-OFF}
build_openssl=${LLAMA_CPP_OPENSSL:-OFF}

need_git_lfs=0
need_jinja2=0

if [ -z "${ctest_regex}" ] || [[ "${ctest_regex}" =~ test-tokenizers-ggml-vocabs ]]; then
    need_git_lfs=1
fi

if [ -z "${ctest_regex}" ] || [[ "${ctest_regex}" =~ test-jinja-py ]]; then
    need_jinja2=1
fi

if ((need_jinja2)) && ! python3 -c 'import jinja2' >/dev/null 2>&1; then
    pip_args=(install --no-cache-dir Jinja2)

    if [ -n "${python_target_dir}" ]; then
        mkdir -p "${python_target_dir}"
        export PYTHONPATH="${python_target_dir}${PYTHONPATH:+:${PYTHONPATH}}"
        pip_args=(install --no-cache-dir --target "${python_target_dir}" Jinja2)
    fi

    python3 -m pip "${pip_args[@]}"
fi

if ((need_git_lfs)); then
    bash "${src_dir}/scripts/prepare-tokenizer-vocabs.sh" "${src_dir}"
fi

cuda_architectures=${LLAMA_CPP_CUDA_ARCHITECTURES:-}
if [ -z "${cuda_architectures}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_architectures=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
fi

cmake_args=(
    -S "${src_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=${build_type}
    -DLLAMA_BUILD_TESTS=ON
    -DLLAMA_BUILD_EXAMPLES=${build_examples}
    -DLLAMA_BUILD_TOOLS=${build_tools}
    -DLLAMA_BUILD_SERVER=${build_server}
    -DLLAMA_OPENSSL=${build_openssl}
    -DGGML_CUDA=ON
    -DGGML_CUDA_CUB_3DOT2=ON
)

if [ -n "${cuda_architectures}" ]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${cuda_architectures}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel "${build_jobs}"

ctest_args=(--test-dir "${build_dir}" --output-on-failure)
if [ -n "${ctest_regex}" ]; then
    ctest_args+=(-R "${ctest_regex}")
fi

ctest "${ctest_args[@]}"