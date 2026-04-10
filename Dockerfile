ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        curl \
        git \
        libgomp1 \
        libssl-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete \
    && git config --system --add safe.directory /workspace

ENV CMAKE_GENERATOR=Ninja \
    CCACHE_DIR=/ccache \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

CMD ["/bin/bash"]