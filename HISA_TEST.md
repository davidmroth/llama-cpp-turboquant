From `/media/data/projects/model-runner` on the remote host:

```bash
docker compose --profile test run --rm \
	-v /media/data/projects/llama-cpp-turboquant:/build/llama.cpp \
	ai-test bash -lc 'bash /build/llama.cpp/cuda-ctest.sh'
```

This defaults to the CUDA-relevant test subset:

- `test-backend-ops`
- `test-quantize-fns`

Optional overrides while iterating:

```bash
docker compose --profile test run --rm \
	-v /media/data/projects/llama-cpp-turboquant:/build/llama.cpp \
	ai-test bash -lc 'LLAMA_CPP_CTEST_REGEX="test-backend-ops|test-quantize-fns|test-jinja-py" bash /build/llama.cpp/cuda-ctest.sh'
```

If you want the full suite instead of the CUDA-focused subset, clear the regex:

```bash
docker compose --profile test run --rm \
	-v /media/data/projects/llama-cpp-turboquant:/build/llama.cpp \
	ai-test bash -lc 'LLAMA_CPP_CTEST_REGEX= bash /build/llama.cpp/cuda-ctest.sh'
```

Note: the full suite may still require `git-lfs` inside the container for `test-tokenizers-ggml-vocabs`.

If the mounted checkout is stale and it is a real git repo, update it first:

```bash
cd /media/data/projects/llama-cpp-turboquant && git pull
```