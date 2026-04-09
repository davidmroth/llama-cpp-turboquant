#!/bin/bash
# V3 next experiments:
# 1) TQ + V3 90% (prefix=128) full stack on 7B @ 32K
# 2) V3 @ 90% (prefix=128) transfer to Qwen3.5-27B and Qwen3.5-35B-A3B @ 32K
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
M_27B="$MODELS_DIR/Qwen3.5-27B-Q8_0.gguf"
M_35B="$MODELS_DIR/Qwen3.5-35B-A3B-Q8_0.gguf"
RESULTS="$LLAMA_DIR/v3_stack_transfer_results.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

run() {
  local name=$1 model=$2 budget=$3 hybrid=$4 prefix=$5 extra=$6
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"
  log ">>> $name (budget=$budget, hybrid=$hybrid, prefix=$prefix, extra=$extra)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=$hybrid TRIATT_PREFIX=$prefix "$BENCH" -m "$model" -f "$WIKI" -c 32768 -b 512 --chunks 3 $triatt $extra 2>&1)
  local rt=$(($(date +%s) - start))
  local ppl=$(echo "$out" | grep "Final estimate" | tail -1)
  local evs=$(echo "$out" | grep -c "evict: evicted")
  log "  $ppl"
  log "  evict_rounds=$evs runtime=${rt}s"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "V3 stack + transfer"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

# -----------------------------------------------
# 1) TQ + V3 90% full stack on 7B @ 32K
# -----------------------------------------------
log ""
log "========== 1) TQ + V3 90% full stack (7B @ 32K) =========="
run "7B TQ only (reference)"         "$M_7B" 0     0 256 "-ctk q8_0 -ctv turbo3"
run "7B V3 90% only (reference)"     "$M_7B" 29491 2 128 ""
run "7B TQ + V3 90% STACK"           "$M_7B" 29491 2 128 "-ctk q8_0 -ctv turbo3"

# -----------------------------------------------
# 2) V3 90% transfer to 27B and 35B-A3B
# -----------------------------------------------
log ""
log "========== 2) V3 @ 90% transfer to qwen35 hybrid models =========="
run "27B V1 90% (reference)" "$M_27B" 29491 0 128 ""
run "27B V3 90%"             "$M_27B" 29491 2 128 ""

run "35B V1 90% (reference)" "$M_35B" 29491 0 128 ""
run "35B V3 90%"             "$M_35B" 29491 2 128 ""

echo "============================================"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================"
