#!/usr/bin/env python3
"""Convert TriAttention calibration stats (.pt) to llama.cpp binary format (.triatt).

TriAttention: Trigonometric KV Cache Token Eviction
Based on "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"
by Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen
arXiv:2604.04921, 2026 — Apache 2.0 License
Original source: https://github.com/WeianMao/triattention

Usage:
    # 1. Generate stats using upstream TriAttention (pinned version):
    pip install git+https://github.com/WeianMao/triattention@main
    python -m triattention.scripts.calibrate \
        --model Qwen/Qwen2-7B \
        --input calibration_text.txt \
        --output qwen2_stats.pt

    # Or use their standalone script:
    python triattention/scripts/calibrate.py \
        --model Qwen/Qwen2-7B \
        --input calibration_text.txt \
        --output qwen2_stats.pt

    # 2. Convert to llama.cpp format:
    python3 scripts/convert_triatt_stats.py \
        --input qwen2_stats.pt \
        --output qwen2.triatt \
        --rope-theta 1000000.0 \
        --num-kv-heads 4
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
import numpy as np


MAGIC = 0x54524941  # "TRIA"
VERSION = 1


def convert(
    input_path: str,
    output_path: str,
    rope_theta: float | None = None,
    num_kv_heads: int | None = None,
) -> None:
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    metadata = data.get("metadata", {})
    stats = data.get("stats", {})

    if not stats:
        print("ERROR: No stats found in .pt file", file=sys.stderr)
        sys.exit(1)

    # Parse layer/head structure from keys like "layer00_head00"
    layers_heads: dict[int, list[int]] = {}
    for key in sorted(stats.keys()):
        parts = key.split("_")
        layer_idx = int(parts[0].replace("layer", ""))
        head_idx = int(parts[1].replace("head", ""))
        layers_heads.setdefault(layer_idx, []).append(head_idx)

    n_layers = len(layers_heads)
    n_attn_heads = max(len(heads) for heads in layers_heads.values())

    # Determine head_dim from first entry
    first_key = sorted(stats.keys())[0]
    freq_count = len(stats[first_key]["q_mean_real"])
    head_dim = freq_count * 2

    # Determine n_kv_heads (GQA)
    if num_kv_heads is not None:
        n_kv_heads = num_kv_heads
    else:
        n_kv_heads = metadata.get("num_kv_heads", n_attn_heads)

    # Determine rope_theta
    if rope_theta is None:
        rope_theta = metadata.get("rope_theta", 10000.0)

    # Determine rope_style
    rope_style_str = metadata.get("rope_style", "half")
    rope_style = 0 if rope_style_str == "half" else 1

    print(f"Model: {n_layers} layers, {n_attn_heads} attn heads, {n_kv_heads} KV heads", file=sys.stderr)
    print(f"head_dim={head_dim}, freq_count={freq_count}", file=sys.stderr)
    print(f"rope_theta={rope_theta}, rope_style={rope_style_str}", file=sys.stderr)

    # GQA: average attention head stats down to KV head count
    heads_per_kv = n_attn_heads // n_kv_heads

    with open(output_path, "wb") as f:
        # Write header (32 bytes)
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", n_layers))
        f.write(struct.pack("<I", n_kv_heads))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", freq_count))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<I", rope_style))

        # Write per-layer, per-kv-head stats
        for layer_idx in range(n_layers):
            for kv_head_idx in range(n_kv_heads):
                # Average over attention heads that map to this KV head
                q_mean_real_acc = np.zeros(freq_count, dtype=np.float32)
                q_mean_imag_acc = np.zeros(freq_count, dtype=np.float32)
                q_abs_mean_acc = np.zeros(freq_count, dtype=np.float32)

                for g in range(heads_per_kv):
                    attn_head = kv_head_idx * heads_per_kv + g
                    key = f"layer{layer_idx:02d}_head{attn_head:02d}"
                    if key not in stats:
                        print(f"  [warn] Missing stats for {key}", file=sys.stderr)
                        continue
                    entry = stats[key]
                    q_mean_real_acc += entry["q_mean_real"].numpy().astype(np.float32)
                    q_mean_imag_acc += entry["q_mean_imag"].numpy().astype(np.float32)
                    q_abs_mean_acc += entry["q_abs_mean"].numpy().astype(np.float32)

                q_mean_real_acc /= heads_per_kv
                q_mean_imag_acc /= heads_per_kv
                q_abs_mean_acc /= heads_per_kv

                f.write(q_mean_real_acc.tobytes())
                f.write(q_mean_imag_acc.tobytes())
                f.write(q_abs_mean_acc.tobytes())

    file_size = Path(output_path).stat().st_size
    print(f"Wrote {output_path} ({file_size} bytes, {n_layers * n_kv_heads} head blocks)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TriAttention .pt stats to llama.cpp .triatt binary format."
    )
    parser.add_argument("--input", required=True, help="Input .pt stats file from TriAttention calibration")
    parser.add_argument("--output", required=True, help="Output .triatt binary file")
    parser.add_argument("--rope-theta", type=float, default=None,
                        help="RoPE theta (auto-detected from stats if available)")
    parser.add_argument("--num-kv-heads", type=int, default=None,
                        help="Number of KV heads for GQA models (auto-detected if available)")
    args = parser.parse_args()
    convert(args.input, args.output, args.rope_theta, args.num_kv_heads)


if __name__ == "__main__":
    main()
