#!/bin/bash
# V3 critical ablations on Qwen2.5-7B @ 32K:
# 1) V3 @ 85% retention (is the win conservative-only?)
# 2) V3 @ 90% with prefix=128 vs prefix=256 (is 128 a tighter default?)
# Each run produces 3-chunk PPL + strict NIAH start/middle/end.
set +e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
RESULTS="$LLAMA_DIR/v3_ablations_results.txt"
TMP_PROMPT="/tmp/v3_ablations_prompt.txt"

> "$RESULTS"

log() {
  echo "$(date '+%H:%M:%S') $*" | tee -a "$RESULTS"
}

run_ppl() {
  local name=$1 budget=$2 prefix=$3
  log ">>> PPL $name (budget=$budget, V3, prefix=$prefix)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=2 TRIATT_PREFIX=$prefix "$BENCH" -m "$M_7B" -f "$WIKI" -c 32768 -b 512 --chunks 3 --triatt-budget $budget 2>&1)
  local rt=$(($(date +%s) - start))
  local ppl=$(echo "$out" | grep "Final estimate" | tail -1)
  local evs=$(echo "$out" | grep -c "evict: evicted")
  log "  $ppl"
  log "  evict_rounds=$evs runtime=${rt}s"
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

run_niah() {
  local name=$1 budget=$2 prefix=$3 pos_chars=$4
  build_prompt "$pos_chars"
  log ">>> NIAH $name @$pos_chars (budget=$budget, V3 prefix=$prefix)"
  local start=$(date +%s)
  local out
  out=$(TRIATT_HYBRID=2 TRIATT_PREFIX=$prefix "$COMP" -m "$M_7B" -f "$TMP_PROMPT" -n 32 -c 32768 --temp 0 -no-cnv --no-display-prompt --triatt-budget $budget 2>/tmp/v3_ablations_stderr.txt)
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
  evs=$(grep -c "evict: evicted" /tmp/v3_ablations_stderr.txt 2>/dev/null | head -1)
  log "  $result (runtime=${rt}s, evictions=$evs)"
  log "  generated: $(echo "$gen" | tr '\n' ' ' | cut -c1-200)"
  echo "" >> "$RESULTS"
}

echo "============================================"
echo "V3 Ablations — 7B @ 32K"
echo "Started: $(date)"
echo "============================================" | tee -a "$RESULTS"

# -----------------------------------------------
# A) V3 @ 85% with prefix=256 (the known-good prefix)
# -----------------------------------------------
log ""
log "========== A) V3 @ 85% retention (prefix=256) =========="
run_ppl "V3-85 prefix=256" 27853 256
run_niah "V3-85 prefix=256" 27853 256 400
run_niah "V3-85 prefix=256" 27853 256 65000
run_niah "V3-85 prefix=256" 27853 256 120000

# -----------------------------------------------
# B) V3 @ 90% with prefix=128 (smaller prefix ablation)
# -----------------------------------------------
log ""
log "========== B) V3 @ 90% retention, prefix=128 =========="
run_ppl "V3-90 prefix=128" 29491 128
run_niah "V3-90 prefix=128" 29491 128 400
run_niah "V3-90 prefix=128" 29491 128 65000
run_niah "V3-90 prefix=128" 29491 128 120000

echo "============================================"
echo "Done: $(date)" | tee -a "$RESULTS"
echo "============================================"
