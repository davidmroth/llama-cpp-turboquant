#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <export-graph-ops> <model>" >&2
    exit 1
fi

export_graph_ops="$1"
model="$2"
log_file="$(mktemp)"
out_file="$(mktemp)"
trap 'rm -f "$log_file" "$out_file"' EXIT

if "$export_graph_ops" \
    -m "$model" \
    -c 128 \
    -b 128 \
    -ub 128 \
    -ctk q8_0 \
    -ctv turbo4 \
    -o "$out_file" \
    > /dev/null 2>"$log_file"; then
    echo "expected export-graph-ops to reject invalid cache configuration" >&2
    exit 1
fi

grep -q 'failed to create llama_context' "$log_file"