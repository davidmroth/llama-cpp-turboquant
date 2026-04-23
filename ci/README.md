# CI

This CI implements heavy-duty workflows that run on self-hosted runners. Typically the purpose of these workflows is to
cover hardware configurations that are not available from Github-hosted runners and/or require more computational
resource than normally available.

It is a good practice, before publishing changes to execute the full CI locally on your machine. For example:

```bash
mkdir tmp

# CPU-only build
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with CUDA support
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# compile-only CUDA build without a visible NVIDIA GPU
GG_BUILD_CUDA=1 GG_BUILD_ONLY=1 GG_BUILD_CUDA_ARCHITECTURES="75;80;86;89" bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with SYCL support
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# with MUSA support
GG_BUILD_MUSA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# etc.
```

# Adding self-hosted runners

- Add a self-hosted `ggml-ci` workflow to [[.github/workflows/build.yml]] with an appropriate label
- Request a runner token from `ggml-org` (for example, via a comment in the PR or email)
- Set-up a machine using the received token ([docs](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners))
- Optionally update [ci/run.sh](https://github.com/ggml-org/llama.cpp/blob/master/ci/run.sh) to build and run on the target platform by gating the implementation with a `GG_BUILD_...` env

For CUDA specifically, there are two distinct validation tiers:

- A host without a visible NVIDIA GPU can still do a compile-only CUDA check by setting `GG_BUILD_ONLY=1` and an explicit `GG_BUILD_CUDA_ARCHITECTURES` list.
- Real CUDA runtime testing still requires a Linux runner with the NVIDIA driver, container toolkit, and accessible GPU device(s).
