#!/bin/bash
# Fixed NIAH script for 7B: uses -no-cnv, prompt file, exact needle match
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
RESULTS="$LLAMA_DIR/niah_7b_strict_results.txt"
TMP_PROMPT="/tmp/niah_7b_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

build_prompt() {
  local ctx=$1 needle_pos=$2
  local needle="The secret code word is PURPLE ELEPHANT 7742."
  local question=" What is the secret code word mentioned earlier? Answer with just the code word and number, nothing else:"
  local total_chars=$((ctx * 4))

  local wiki_text
  wiki_text=$(cat "$WIKI")

  local before="${wiki_text:0:$needle_pos}"
  local after_start=$needle_pos
  local after_len=$((total_chars - needle_pos - ${#needle} - ${#question} - 50))
  [ $after_len -lt 100 ] && after_len=100
  local after="${wiki_text:$after_start:$after_len}"

  printf "%s %s %s %s" "$before" "$needle" "$after" "$question" > "$TMP_PROMPT"
}

run_niah() {
  local name=$1 ctx=$2 budget=$3 pos_chars=$4
  local triatt=""
  [ "$budget" -gt 0 ] && triatt="--triatt-budget $budget"

  build_prompt "$ctx" "$pos_chars"

  log ">>> $name (c=$ctx, budget=$budget, needle@$pos_chars chars)"
  local start=$(date +%s)
  local out
  # --no-display-prompt: only print generated tokens, not the echoed prompt.
  # CRITICAL: without this, the prompt (which contains the needle) gets echoed
  # to stdout and gives a false PASS for every test.
  out=$("$COMP" -m "$M_7B" -f "$TMP_PROMPT" -n 32 -c "$ctx" --temp 0 -no-cnv --no-display-prompt $triatt 2>/tmp/niah_7b_stderr.txt)
  local rt=$(($(date +%s) - start))

  # Strip llama.cpp's perf print and any [end of text] tokens
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
  evs=$(grep -c "evict:" /tmp/niah_7b_stderr.txt 2>/dev/null || echo 0)
  evs=$(echo "$evs" | head -1)

  log "  $result (runtime=${rt}s, evictions=$evs)"
  log "  generated: $(echo "$gen" | tr '\n' ' ' | cut -c1-200)"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "7B NIAH Strict Re-run: $(date)"
echo "============================================" | tee -a "$RESULTS"

log ""
log "========== BASELINE (100%) =========="
run_niah "baseline start"  32768 0 400
run_niah "baseline middle" 32768 0 65000
run_niah "baseline end"    32768 0 120000

log ""
log "========== TRIATT 90% =========="
run_niah "90% start"  32768 29491 400
run_niah "90% middle" 32768 29491 65000
run_niah "90% end"    32768 29491 120000

log ""
log "========== TRIATT 85% =========="
run_niah "85% start"  32768 27853 400
run_niah "85% middle" 32768 27853 65000
run_niah "85% end"    32768 27853 120000

echo "============================================"
echo "NIAH complete: $(date)" | tee -a "$RESULTS"
echo "============================================"
