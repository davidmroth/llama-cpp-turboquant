#include "../src/llama-graph.h"

#include <cassert>

int main() {
    assert(llama_hisa_supports_kv_types(GGML_TYPE_F16,     GGML_TYPE_F16));
    assert(llama_hisa_supports_kv_types(GGML_TYPE_Q8_0,    GGML_TYPE_F16));
    assert(llama_hisa_supports_kv_types(GGML_TYPE_F16,     GGML_TYPE_TURBO4_0));
    assert(llama_hisa_supports_kv_types(GGML_TYPE_Q8_0,    GGML_TYPE_TURBO4_0));

    assert(!llama_hisa_supports_kv_types(GGML_TYPE_TURBO2_0, GGML_TYPE_F16));
    assert(!llama_hisa_supports_kv_types(GGML_TYPE_TURBO3_0, GGML_TYPE_F16));
    assert(!llama_hisa_supports_kv_types(GGML_TYPE_TURBO4_0, GGML_TYPE_F16));
    assert(!llama_hisa_supports_kv_types(GGML_TYPE_F16,      GGML_TYPE_TURBO2_0));
    assert(!llama_hisa_supports_kv_types(GGML_TYPE_F16,      GGML_TYPE_TURBO3_0));

    return 0;
}