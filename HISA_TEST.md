apt-get update && apt-get install -y git-lfs python3-jinja2
git lfs install
cd /build/llama.cpp/models/ggml-vocabs
git lfs pull
cmake -S /build/llama.cpp/ -B /building -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /building/ --parallel
ctest --test-dir /building --output-on-failure