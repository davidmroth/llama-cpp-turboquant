#!/usr/bin/env bash

set -euo pipefail

repo_dir=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
target_dir=${LLAMA_CPP_VOCABS_DIR:-${repo_dir}/models/ggml-vocabs}
repo_url=${LLAMA_CPP_VOCABS_REPO:-https://huggingface.co/ggml-org/vocabs}

ensure_git_lfs() {
    if command -v git-lfs >/dev/null 2>&1; then
        return
    fi

    local os arch platform version cache_dir install_dir bin_dir archive tmp_dir extracted_dir

    os=$(uname -s)
    arch=$(uname -m)

    case "${os}/${arch}" in
        Linux/x86_64)
            platform=linux-amd64
            ;;
        Linux/aarch64|Linux/arm64)
            platform=linux-arm64
            ;;
        Darwin/x86_64)
            platform=darwin-amd64
            ;;
        Darwin/arm64)
            platform=darwin-arm64
            ;;
        *)
            echo "Unsupported platform for automatic git-lfs bootstrap: ${os}/${arch}" >&2
            exit 1
            ;;
    esac

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to bootstrap git-lfs automatically." >&2
        exit 1
    fi

    version=${LLAMA_CPP_GIT_LFS_VERSION:-3.7.0}
    cache_dir=${LLAMA_CPP_GIT_LFS_CACHE_DIR:-${HOME}/.cache/llama.cpp}
    install_dir=${LLAMA_CPP_GIT_LFS_INSTALL_DIR:-${HOME}/.local/opt/git-lfs/v${version}}
    bin_dir=${LLAMA_CPP_GIT_LFS_BIN_DIR:-${HOME}/.local/bin}
    archive=${cache_dir}/git-lfs-${platform}-v${version}.tar.gz

    mkdir -p "${cache_dir}" "${install_dir}" "${bin_dir}"

    if [ ! -f "${archive}" ]; then
        curl -fsSL -o "${archive}" "https://github.com/git-lfs/git-lfs/releases/download/v${version}/git-lfs-${platform}-v${version}.tar.gz"
    fi

    if [ ! -x "${install_dir}/git-lfs" ]; then
        tmp_dir=$(mktemp -d)
        tar -xzf "${archive}" -C "${tmp_dir}"
        extracted_dir=$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)

        if [ -z "${extracted_dir}" ] || [ ! -x "${extracted_dir}/git-lfs" ]; then
            echo "Failed to unpack git-lfs from ${archive}" >&2
            rm -rf "${tmp_dir}"
            exit 1
        fi

        install -m 0755 "${extracted_dir}/git-lfs" "${install_dir}/git-lfs"
        rm -rf "${tmp_dir}"
    fi

    ln -sf "${install_dir}/git-lfs" "${bin_dir}/git-lfs"
    export PATH="${bin_dir}:${install_dir}:${PATH}"
}

ensure_git_lfs

if [ -d "${target_dir}/.git" ]; then
    git -C "${target_dir}" pull --ff-only
else
    rm -rf "${target_dir}"
    git clone "${repo_url}" "${target_dir}"
fi

git -C "${target_dir}" lfs install --local
git -C "${target_dir}" lfs pull

if ! find "${target_dir}" -type f -name '*.gguf' -print -quit | grep -q .; then
    echo "No GGUF files found under ${target_dir} after git-lfs pull." >&2
    exit 1
fi