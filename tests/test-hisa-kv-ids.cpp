#include "../src/llama-graph.h"

#include "ggml.h"

#include <cassert>

static void test_hisa_kv_ids_for_layout(int64_t n_head_kv, int64_t n_stream) {
    ggml_init_params params = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * k_perm   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 1024, n_head_kv, n_stream);
    ggml_tensor * mask_ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 128, 1, n_stream, 1);
    ggml_tensor * kv_ids   = llama_hisa_expand_kv_ids(ctx, mask_ids, k_perm);

    assert(kv_ids->type == GGML_TYPE_I32);
    assert(kv_ids->ne[0] == mask_ids->ne[0]);
    assert(kv_ids->ne[1] == n_head_kv);
    assert(kv_ids->ne[2] == n_stream);
    assert(kv_ids->ne[3] == 1);

    ggml_tensor * gathered = ggml_get_rows(ctx, k_perm, kv_ids);
    assert(gathered->ne[0] == k_perm->ne[0]);
    assert(gathered->ne[1] == kv_ids->ne[0]);
    assert(gathered->ne[2] == kv_ids->ne[1]);
    assert(gathered->ne[3] == kv_ids->ne[2]);

    ggml_free(ctx);
}

int main() {
    test_hisa_kv_ids_for_layout(2, 1);
    test_hisa_kv_ids_for_layout(8, 3);
    return 0;
}