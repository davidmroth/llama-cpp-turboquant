#!/bin/bash
set +e  # keep going even if individual tests fail

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
RESULTS="$LLAMA_DIR/followup_results.txt"

M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

run_ppl() {
  local name=$1 ctx=$2 budget=$3 extra_args=$4
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  log ">>> $name (c=$ctx, budget=$budget) $extra_args"
  local start=$(date +%s)
  local out=$("$BENCH" -m "$M_7B" -f "$WIKI" -c "$ctx" -b 512 --chunks 3 $triatt $extra_args 2>&1)
  local rt=$(($(date +%s) - start))
  local ppl=$(echo "$out" | grep "Final" | tail -1)
  local evs=$(echo "$out" | grep -c "evict:")
  log "  $ppl"
  log "  evictions=$evs runtime=${rt}s"
  echo "" >> "$RESULTS"
}

run_niah_strict() {
  local name=$1 ctx=$2 budget=$3 pos_chars=$4
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  local needle="The secret code word is PURPLE ELEPHANT 7742."
  local wiki_text=$(cat "$WIKI")
  local question=" What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:"
  local total_chars=$((ctx * 4))

  local before="${wiki_text:0:$pos_chars}"
  local after_start=$pos_chars
  local after_len=$((total_chars - pos_chars - ${#needle} - ${#question} - 50))
  [ $after_len -lt 0 ] && after_len=100
  local after="${wiki_text:$after_start:$after_len}"
  local prompt="$before $needle $after $question"

  log ">>> NIAH $name (c=$ctx, budget=$budget, needle@$pos_chars)"
  local start=$(date +%s)
  local out=$("$COMP" -m "$M_7B" -p "$prompt" -n 32 -c "$ctx" --temp 0 $triatt 2>/dev/null < /dev/null | tail -3)
  local rt=$(($(date +%s) - start))

  local result="FAIL"
  if echo "$out" | grep -qi "PURPLE ELEPHANT 7742"; then
    result="PASS"
  elif echo "$out" | grep -qi "PURPLE ELEPHANT"; then
    result="PARTIAL_WORD"
  elif echo "$out" | grep -qi "7742"; then
    result="PARTIAL_NUMBER"
  fi

  log "  $result (runtime=${rt}s)"
  log "  output: $(echo "$out" | head -1 | cut -c1-120)"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "TriAttention Follow-Up Tests"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

# ============================================
# TASK 1: Replicate 7B @ 32K, 90% on 3 chunks
# ============================================
log ""
log "========== TASK 1: 7B @ 32K Replication (3 chunks) =========="
run_ppl "7B 32K baseline" 32768 0
run_ppl "7B 32K 90%" 32768 29491
run_ppl "7B 32K 85%" 32768 27853

# ============================================
# TASK 2: 7B @ 64K baseline vs 90% vs 85%
# ============================================
log ""
log "========== TASK 2: 7B @ 64K =========="
run_ppl "7B 64K baseline" 65536 0
run_ppl "7B 64K 90%" 65536 58982
run_ppl "7B 64K 85%" 65536 55706

# ============================================
# TASK 3: NIAH with strict checker
# ============================================
log ""
log "========== TASK 3: NIAH 7B @ 32K (strict checker) =========="

# Baseline
run_niah_strict "baseline start" 32768 0 400
run_niah_strict "baseline middle" 32768 0 65000
run_niah_strict "baseline end" 32768 0 120000

# 90%
run_niah_strict "90% start" 32768 29491 400
run_niah_strict "90% middle" 32768 29491 65000
run_niah_strict "90% end" 32768 29491 120000

# 85%
run_niah_strict "85% start" 32768 27853 400
run_niah_strict "85% middle" 32768 27853 65000
run_niah_strict "85% end" 32768 27853 120000

# ============================================
# TASK 4: Full Stack at 32K (f16 / TQ / TriAtt / stack)
# ============================================
log ""
log "========== TASK 4: Full Stack @ 32K =========="
run_ppl "f16 KV baseline" 32768 0
run_ppl "TQ only (q8_0 + turbo3)" 32768 0 "-ctk q8_0 -ctv turbo3"
run_ppl "TriAtt 90% only" 32768 29491
run_ppl "TQ + TriAtt 90% (STACK)" 32768 29491 "-ctk q8_0 -ctv turbo3"

echo "============================================"
echo "ALL FOLLOW-UP TESTS COMPLETE: $(date)" | tee -a "$RESULTS"
echo "Results: $RESULTS"
echo "============================================"
