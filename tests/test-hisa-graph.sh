#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <export-graph-ops> <model> <output-file>" >&2
    exit 1
fi

export_graph_ops="$1"
model="$2"
output_file="$3"

rm -f "$output_file"

"$export_graph_ops" \
    -m "$model" \
    -c 4096 \
    -b 4096 \
    -ub 4096 \
    --prefill-attn hisa \
    --hisa-top-k 256 \
    --hisa-block-size 128 \
    --hisa-top-m-blocks 8 \
    --hisa-min-seq-len 2048 \
    --hisa-local-window 512 \
    --hisa-reuse-ratio 0.75 \
    -o "$output_file" \
    >/dev/null

grep -q 'hisa_q_summary' "$output_file"
grep -q 'hisa_top_blocks' "$output_file"
grep -q 'hisa_k_sparse' "$output_file"
grep -q 'hisa_reused_blocks' "$output_file"
grep -q 'hisa_block_scores_masked' "$output_file"
grep -q 'hisa_top_blocks_refresh' "$output_file"