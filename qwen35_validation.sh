#!/bin/bash
# Qwen3.5-27B and Qwen3.5-35B-A3B PPL validation at 32K with TriAttention.
# Goal: confirm eviction fires (eviction_rounds > 0) and produces a valid PPL.
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
M_27B="$MODELS_DIR/Qwen3.5-27B-Q8_0.gguf"
M_35B="$MODELS_DIR/Qwen3.5-35B-A3B-Q8_0.gguf"
RESULTS="$LLAMA_DIR/qwen35_validation_results.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

run() {
  local name=$1 model=$2 ctx=$3 budget=$4 chunks=$5
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  log ">>> $name (c=$ctx, budget=$budget, chunks=$chunks)"
  local start=$(date +%s)
  local out
  out=$("$BENCH" -m "$model" -f "$WIKI" -c "$ctx" -b 512 --chunks "$chunks" $triatt 2>&1)
  local rt=$(($(date +%s) - start))

  local ppl=$(echo "$out" | grep "Final estimate" | tail -1)
  local evs=$(echo "$out" | grep -c "evict: evicted")
  local cal=$(echo "$out" | grep -c "update_calibration")

  log "  $ppl"
  log "  evict_rounds=$evs calibrations=$cal runtime=${rt}s"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "Qwen3.5 27B/35B-A3B validation at 32K"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

log ""
log "========== Qwen3.5-27B @ 32K =========="
run "27B baseline" "$M_27B" 32768 0 3
run "27B 90%"     "$M_27B" 32768 29491 3
run "27B 85%"     "$M_27B" 32768 27853 3

log ""
log "========== Qwen3.5-35B-A3B @ 32K =========="
run "35B baseline" "$M_35B" 32768 0 3
run "35B 90%"      "$M_35B" 32768 29491 3
run "35B 85%"      "$M_35B" 32768 27853 3

echo "============================================"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================"
