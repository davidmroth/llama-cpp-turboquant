#!/bin/bash
# SMEM pre-dequant experiment benchmarks
# Run on Mac Mini M2 Pro after 7pm CST (DINOv2 training finishes)
# Usage: ssh toms-mac-mini.local "cd ~/dev/turbo_test/llama-cpp-turbo && bash bench_smem_experiment.sh"

MODEL=~/dev/turbo_test/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
BENCH=./build/bin/llama-bench
COMMON="-ngl 99 -fa 1 -t 1 -n 128"

echo "=== SMEM Pre-Dequant Experiment Benchmarks ==="
echo "Date: $(date)"
echo ""

echo "--- q8_0 baseline (8K) ---"
$BENCH -m $MODEL $COMMON -ctk q8_0 -ctv q8_0 -p 8192

echo ""
echo "--- turbo3 4-mag baseline (short) ---"
$BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 0

echo ""
echo "--- turbo3 4-mag baseline (8K) ---"
$BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 8192

echo ""
echo "--- turbo3 4-mag baseline (16K) ---"
$BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 16384

echo ""
echo "--- turbo3 SMEM pre-dequant (8K) ---"
TURBO_SMEM_DEQUANT=1 $BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 8192

echo ""
echo "--- turbo3 QC precompute (8K) ---"
TURBO_QC_PRECOMPUTE=1 $BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 8192

echo ""
echo "--- turbo3 no-dequant ceiling (8K) ---"
TURBO_PROFILE_MODE=1 $BENCH -m $MODEL $COMMON -ctk turbo3 -ctv turbo3 -p 8192

echo ""
echo "=== Done ==="
echo "Date: $(date)"
