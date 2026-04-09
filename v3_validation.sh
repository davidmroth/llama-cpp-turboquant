#!/bin/bash
# V3 (prefix_protect + per-segment quota) validation on Qwen2.5-7B @ 32K, 90% only.
# Test: 3-chunk PPL + strict NIAH at start/middle/end.
# Compare against baseline (already measured), V1 90% (already measured), V2 90% (already measured).
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
RESULTS="$LLAMA_DIR/v3_validation_results.txt"
TMP_PROMPT="/tmp/v3_niah_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

run_ppl_v3() {
  log ">>> PPL V3 hybrid 90% (budget=29491, TRIATT_HYBRID=2, prefix=256)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=2 "$BENCH" -m "$M_7B" -f "$WIKI" -c 32768 -b 512 --chunks 3 --triatt-budget 29491 2>&1)
  local rt=$(($(date +%s) - start))
  local ppl=$(echo "$out" | grep "Final estimate" | tail -1)
  local evs=$(echo "$out" | grep -c "evict: evicted")
  local mode=$(echo "$out" | grep "evict: evicted" | head -1 | grep -oE "mode=[a-z0-9]+" | head -1)
  log "  $ppl"
  log "  evict_rounds=$evs $mode runtime=${rt}s"
  echo "" >> "$RESULTS"
}

build_prompt() {
  local pos_chars=$1
  local needle="The secret code word is PURPLE ELEPHANT 7742."
  local question=" What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:"
  local total_chars=$((32768 * 4))
  local wiki_text
  wiki_text=$(cat "$WIKI")
  local before="${wiki_text:0:$pos_chars}"
  local after_start=$pos_chars
  local after_len=$((total_chars - pos_chars - ${#needle} - ${#question} - 50))
  [ $after_len -lt 100 ] && after_len=100
  local after="${wiki_text:$after_start:$after_len}"
  printf "%s %s %s %s" "$before" "$needle" "$after" "$question" > "$TMP_PROMPT"
}

run_niah_v3() {
  local pos_chars=$1
  build_prompt "$pos_chars"
  log ">>> NIAH V3 hybrid 90% @$pos_chars (budget=29491, TRIATT_HYBRID=2, prefix=256)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=2 "$COMP" -m "$M_7B" -f "$TMP_PROMPT" -n 32 -c 32768 --temp 0 -no-cnv --no-display-prompt --triatt-budget 29491 2>/tmp/v3_niah_stderr.txt)
  local rt=$(($(date +%s) - start))

  local gen
  gen=$(echo "$out" | grep -v "^common_perf_print\|^llama_memory_breakdown\|^ggml_metal\|^$" | head -10)

  local result="FAIL"
  if echo "$gen" | grep -q "PURPLE ELEPHANT 7742"; then
    result="PASS"
  elif echo "$gen" | grep -qi "PURPLE ELEPHANT"; then
    result="PARTIAL_WORD"
  elif echo "$gen" | grep -qi "7742"; then
    result="PARTIAL_NUMBER"
  fi
  local evs
  evs=$(grep -c "evict: evicted" /tmp/v3_niah_stderr.txt 2>/dev/null | head -1)
  log "  $result (runtime=${rt}s, evictions=$evs)"
  log "  generated: $(echo "$gen" | tr '\n' ' ' | cut -c1-200)"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "V3 Hybrid (prefix=256 + per-segment quota) — 7B @ 32K, 90% only"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

log ""
log "========== TASK 1: PPL @ 32K (3 chunks) =========="
run_ppl_v3

log ""
log "========== TASK 2: NIAH @ 32K (start/middle/end) =========="
run_niah_v3 400
run_niah_v3 65000
run_niah_v3 120000

echo "============================================"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================"
