#include "../src/llama-graph.h"

#include <cassert>

int main() {
    // No previous selection -> no reuse.
    assert(!llama_hisa_can_reuse_blocks(0, 64));
    assert(!llama_hisa_can_reuse_blocks(-1, 64));

    // Equal grids (uniform KV across layers) -> reuse allowed.
    assert(llama_hisa_can_reuse_blocks(64, 64));

    // Previous grid smaller than current (current layer has more candidate
    // blocks, e.g. full-attention layer after a local/SWA-pruned layer) ->
    // reuse allowed; all previous indices are in-range.
    assert(llama_hisa_can_reuse_blocks(32, 64));
    assert(llama_hisa_can_reuse_blocks(1, 64));

    // Previous grid larger than current (e.g. Gemma4 ISWA crossing from a
    // full-attention layer to a local-SWA layer with a smaller candidate
    // window) -> reuse must be refused to avoid out-of-range indices in
    // ggml_set_rows/ggml_get_rows on block_scores.
    assert(!llama_hisa_can_reuse_blocks(65, 64));
    assert(!llama_hisa_can_reuse_blocks(128, 64));

    // Edge: current has no candidate blocks at all.
    assert(!llama_hisa_can_reuse_blocks(1, 0));
    assert(!llama_hisa_can_reuse_blocks(0, 0));

    return 0;
}
