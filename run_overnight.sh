#!/bin/bash
set -e

LLAMA_DIR="/Users/tom/local_llms/llama.cpp"
MODELS_DIR="/Users/tom/local_llms/models"
WIKI="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"
BENCH="$LLAMA_DIR/build-test/bin/llama-perplexity"
COMP="$LLAMA_DIR/build-test/bin/llama-completion"
RESULTS="$LLAMA_DIR/overnight_results.jsonl"

M_7B="$MODELS_DIR/Qwen2.5-7B-Instruct-Q8_0.gguf"
M_27B="$MODELS_DIR/Qwen3.5-27B-Q8_0.gguf"
M_35B="$MODELS_DIR/Qwen3.5-35B-A3B-Q8_0.gguf"

> "$RESULTS"

log_result() {
  local model=$1 ctx=$2 retention=$3 budget=$4 task=$5 baseline_or_triatt=$6
  local ppl=$7 delta=$8 eviction_confirmed=$9 eviction_rounds=${10} runtime=${11} pass_fail=${12}
  echo "{\"model\":\"$model\",\"context\":$ctx,\"retention\":\"$retention\",\"budget\":$budget,\"task\":\"$task\",\"config\":\"$baseline_or_triatt\",\"ppl\":$ppl,\"delta\":\"$delta\",\"eviction_confirmed\":$eviction_confirmed,\"eviction_rounds\":$eviction_rounds,\"runtime_sec\":$runtime,\"pass_fail\":\"$pass_fail\"}" >> "$RESULTS"
}

run_ppl() {
  local model_path=$1 model_name=$2 ctx=$3 budget=$4 retention=$5
  local config="baseline"
  local triatt_args=""
  if [ "$budget" -gt 0 ]; then
    triatt_args="--triatt-budget $budget"
    config="triatt_${retention}"
  fi

  echo "$(date '+%H:%M:%S') PPL: $model_name c=$ctx ret=$retention budget=$budget"
  local start_time=$(date +%s)
  local output=$("$BENCH" -m "$model_path" -f "$WIKI" -c "$ctx" -b 512 --chunks 1 $triatt_args 2>&1)
  local end_time=$(date +%s)
  local runtime=$((end_time - start_time))

  local ppl=$(echo "$output" | grep "Final" | awk '{print $4}')
  local evictions=$(echo "$output" | grep -c "evict:" || true)
  local confirmed=false
  [ "$evictions" -gt 0 ] && confirmed=true

  if [ -z "$ppl" ]; then
    ppl="0"
    echo "  WARNING: no PPL result"
  fi

  echo "  PPL=$ppl evictions=$evictions time=${runtime}s"
  log_result "$model_name" "$ctx" "$retention" "$budget" "ppl" "$config" "$ppl" "" "$confirmed" "$evictions" "$runtime" "n/a"
}

run_niah() {
  local model_path=$1 model_name=$2 ctx=$3 budget=$4 retention=$5 needle_pos=$6
  local config="baseline"
  local triatt_args=""
  if [ "$budget" -gt 0 ]; then
    triatt_args="--triatt-budget $budget"
    config="triatt_${retention}"
  fi

  # Build prompt with needle at specified position
  local needle="The secret code word is PURPLE ELEPHANT 7742."
  local wiki_text=$(cat "$WIKI")
  local char_per_token=4
  local needle_char_pos=$((needle_pos * char_per_token))
  local total_chars=$((ctx * char_per_token))
  local question="What is the secret code word mentioned earlier in this text? Answer with just the code word and number, nothing else:"

  local before="${wiki_text:0:$needle_char_pos}"
  local after_start=$((needle_char_pos))
  local after_len=$((total_chars - needle_char_pos - ${#needle} - ${#question} - 20))
  local after="${wiki_text:$after_start:$after_len}"
  local prompt="$before $needle $after $question"

  echo "$(date '+%H:%M:%S') NIAH: $model_name c=$ctx ret=$retention needle@$needle_pos"
  local start_time=$(date +%s)
  local output=$("$COMP" -m "$model_path" -p "$prompt" -n 32 -c "$ctx" --temp 0 $triatt_args 2>/tmp/niah_stderr.txt < /dev/null)
  local end_time=$(date +%s)
  local runtime=$((end_time - start_time))

  local evictions=$(grep -c "evict:" /tmp/niah_stderr.txt 2>/dev/null || true)
  local confirmed=false
  [ "$evictions" -gt 0 ] && confirmed=true

  # Check if answer contains the needle
  local pass="FAIL"
  if echo "$output" | grep -qi "PURPLE ELEPHANT 7742"; then
    pass="PASS"
  elif echo "$output" | grep -qi "PURPLE ELEPHANT"; then
    pass="PARTIAL"
  elif echo "$output" | grep -qi "PURPLE"; then
    pass="PARTIAL"
  fi

  echo "  Result: $pass (evictions=$evictions, time=${runtime}s)"
  echo "  Output: $(echo "$output" | tail -3 | head -1)"
  log_result "$model_name" "$ctx" "$retention" "$budget" "niah_${needle_pos}" "$config" "0" "" "$confirmed" "$evictions" "$runtime" "$pass"
}

echo "============================================"
echo "TriAttention Overnight Validation"
echo "Started: $(date)"
echo "============================================"

# =============================================
# PRIORITY 1: Core PPL at 32K
# =============================================
echo ""
echo "=== PRIORITY 1: PPL at 32K ==="

# 7B Dense
run_ppl "$M_7B" "Qwen2.5-7B" 32768 0 "100%"
run_ppl "$M_7B" "Qwen2.5-7B" 32768 29491 "90%"
run_ppl "$M_7B" "Qwen2.5-7B" 32768 27853 "85%"

# 27B Dense
run_ppl "$M_27B" "Qwen3.5-27B" 32768 0 "100%"
run_ppl "$M_27B" "Qwen3.5-27B" 32768 29491 "90%"
run_ppl "$M_27B" "Qwen3.5-27B" 32768 27853 "85%"

# 35B MoE
run_ppl "$M_35B" "Qwen3.5-35B-A3B" 32768 0 "100%"
run_ppl "$M_35B" "Qwen3.5-35B-A3B" 32768 29491 "90%"
run_ppl "$M_35B" "Qwen3.5-35B-A3B" 32768 27853 "85%"

echo ""
echo "=== PRIORITY 1 COMPLETE ==="
echo ""

# =============================================
# PRIORITY 2: NIAH at 32K
# =============================================
echo "=== PRIORITY 2: NIAH at 32K ==="

for model_name in "Qwen2.5-7B" "Qwen3.5-27B" "Qwen3.5-35B-A3B"; do
  if [ "$model_name" = "Qwen2.5-7B" ]; then model_path="$M_7B"; fi
  if [ "$model_name" = "Qwen3.5-27B" ]; then model_path="$M_27B"; fi
  if [ "$model_name" = "Qwen3.5-35B-A3B" ]; then model_path="$M_35B"; fi

  budget_90=$((32768 * 90 / 100))

  # Baseline NIAH
  run_niah "$model_path" "$model_name" 32768 0 "100%" 100
  run_niah "$model_path" "$model_name" 32768 0 "100%" 16000
  run_niah "$model_path" "$model_name" 32768 0 "100%" 30000

  # TriAtt 90% NIAH
  run_niah "$model_path" "$model_name" 32768 "$budget_90" "90%" 100
  run_niah "$model_path" "$model_name" 32768 "$budget_90" "90%" 16000
  run_niah "$model_path" "$model_name" 32768 "$budget_90" "90%" 30000
done

echo ""
echo "=== PRIORITY 2 COMPLETE ==="
echo ""

# =============================================
# PRIORITY 3: PPL at 64K (27B + 35B only)
# =============================================
echo "=== PRIORITY 3: PPL at 64K ==="

run_ppl "$M_27B" "Qwen3.5-27B" 65536 0 "100%"
run_ppl "$M_27B" "Qwen3.5-27B" 65536 58982 "90%"
run_ppl "$M_27B" "Qwen3.5-27B" 65536 55706 "85%"

run_ppl "$M_35B" "Qwen3.5-35B-A3B" 65536 0 "100%"
run_ppl "$M_35B" "Qwen3.5-35B-A3B" 65536 58982 "90%"
run_ppl "$M_35B" "Qwen3.5-35B-A3B" 65536 55706 "85%"

echo ""
echo "=== PRIORITY 3 COMPLETE ==="
echo ""

echo "============================================"
echo "ALL TESTS COMPLETE: $(date)"
echo "Results: $RESULTS"
echo "============================================"
