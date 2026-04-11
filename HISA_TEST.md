From `/media/data/projects/model-runner` on the remote host:

```bash
mkdir -p /home/bot/llama-cpp-cuda-build /home/bot/llama-cpp-cuda-pylibs

docker compose --profile test run --rm \
	-v /home/bot/llama-cpp-turboquant:/build/llama.cpp \
	-v /home/bot/llama-cpp-cuda-build:/building \
	-v /home/bot/llama-cpp-cuda-pylibs:/pylibs \
	ai-test bash -lc 'LLAMA_CPP_PYTHON_TARGET_DIR=/pylibs bash /build/llama.cpp/scripts/cuda-ctest.sh'
```

This defaults to the CUDA-relevant test subset:

- `test-backend-ops`
- `test-quantize-fns`

Optional overrides while iterating:

```bash
docker compose --profile test run --rm \
	-v /home/bot/llama-cpp-turboquant:/build/llama.cpp \
	-v /home/bot/llama-cpp-cuda-build:/building \
	-v /home/bot/llama-cpp-cuda-pylibs:/pylibs \
	ai-test bash -lc 'LLAMA_CPP_PYTHON_TARGET_DIR=/pylibs LLAMA_CPP_CTEST_REGEX="test-backend-ops|test-quantize-fns|test-jinja-py" bash /build/llama.cpp/scripts/cuda-ctest.sh'
```

If you want the full suite instead of the CUDA-focused subset, clear the regex:

```bash
docker compose --profile test run --rm \
	-v /home/bot/llama-cpp-turboquant:/build/llama.cpp \
	-v /home/bot/llama-cpp-cuda-build:/building \
	-v /home/bot/llama-cpp-cuda-pylibs:/pylibs \
	ai-test bash -lc 'LLAMA_CPP_PYTHON_TARGET_DIR=/pylibs LLAMA_CPP_CTEST_REGEX= bash /build/llama.cpp/scripts/cuda-ctest.sh'
```

When the regex includes `test-tokenizers-ggml-vocabs`, the helper now bootstraps a user-local `git-lfs` binary if needed and prepares `models/ggml-vocabs` before running `ctest`.

If the mounted checkout is stale and it is a real git repo, update it first:

```bash
cd /home/bot/llama-cpp-turboquant && git pull
```