#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'USAGE'
Usage: scripts/hisa_turboquant_report.sh <runs.csv>

Optional env thresholds:
  MEM_REDUCTION_MIN=30        # minimum KV memory reduction (%) for turbo vs f16/f16
  GEN_REGRESSION_MAX=10       # maximum generation slowdown (%) for turbo vs f16/f16
  HISA_PREFILL_GAIN_MIN=10    # minimum prefill gain (%) for hisa vs dense at long context
  HISA_MIN_CTX=65536          # minimum context length for HISA gain checks
USAGE
  exit 1
fi

CSV_PATH="$1"
if [[ ! -f "$CSV_PATH" ]]; then
  echo "error: file not found: $CSV_PATH" >&2
  exit 1
fi

MEM_REDUCTION_MIN="${MEM_REDUCTION_MIN:-30}"
GEN_REGRESSION_MAX="${GEN_REGRESSION_MAX:-10}"
HISA_PREFILL_GAIN_MIN="${HISA_PREFILL_GAIN_MIN:-10}"
HISA_MIN_CTX="${HISA_MIN_CTX:-65536}"

awk -F',' '
  NR == 1 {
    for (i = 1; i <= NF; i++) header[i] = $i;
    ncols = NF;
    next;
  }
  NF == 0 { next; }
  {
    printf "{";
    for (i = 1; i <= ncols; i++) {
      val = $i;
      gsub(/\\/, "\\\\", val);
      gsub(/\"/, "\\\"", val);
      printf "\"%s\":\"%s\"", header[i], val;
      if (i < ncols) {
        printf ",";
      }
    }
    print "}";
  }
' "$CSV_PATH" | jq -rs \
  --argjson mem_min "$MEM_REDUCTION_MIN" \
  --argjson gen_reg_max "$GEN_REGRESSION_MAX" \
  --argjson hisa_gain_min "$HISA_PREFILL_GAIN_MIN" \
  --argjson hisa_min_ctx "$HISA_MIN_CTX" '

def tonum: (tonumber? // 0);
def sortnum($arr): ($arr | map(tonumber) | sort);
def quantile($arr; $p):
  ($arr | map(tonumber)) as $v
  | ($v | length) as $n
  | if $n == 0 then null
    else
      ($v | sort) as $s
      | (($p * ($n - 1)) | floor) as $i
      | (($p * ($n - 1)) - $i) as $f
      | if ($i + 1) < $n
        then (($s[$i] * (1 - $f)) + ($s[$i + 1] * $f))
        else $s[$i]
        end
    end;
def median($arr): quantile($arr; 0.5);
def p95($arr): quantile($arr; 0.95);
def fmt($x): if $x == null then "na" else ($x | tostring) end;
def pct($x): if $x == null then "na" else ((($x * 100) | round / 100) | tostring) end;
def key($attn; $k; $v; $ctx): "\($attn)|\($k)|\($v)|\($ctx)";
def key_ctx($attn; $ctx): "\($attn)|\($ctx)";

def cell_stats($group): {
  attn: $group[0].attn_requested,
  cache_k: $group[0].cache_k_requested,
  cache_v: $group[0].cache_v_requested,
  ctx_len: ($group[0].ctx_len | tonum),
  runs: ($group | length),
  prompt_tps_median: median($group | map(.prompt_tps | tonum)),
  gen_tps_median: median($group | map(.generation_tps | tonum)),
  prompt_ms_p95: p95($group | map(.prompt_time_ms | tonum)),
  gen_ms_p95: p95($group | map(.generation_time_ms | tonum)),
  kv_mem_mib_median: median($group | map(.kv_mem_mib | tonum)),
  mismatch_count: ($group | map(select(.attn_requested != .attn_observed or .cache_k_requested != .cache_k_observed or .cache_v_requested != .cache_v_observed)) | length),
  status_fail_count: ($group | map(select((.status | ascii_downcase) != "ok")) | length)
};

(. | map(. + {
  ctx_len: (.ctx_len | tonum),
  prompt_tps: (.prompt_tps | tonum),
  generation_tps: (.generation_tps | tonum),
  prompt_time_ms: (.prompt_time_ms | tonum),
  generation_time_ms: (.generation_time_ms | tonum),
  kv_mem_mib: (.kv_mem_mib | tonum),
  status: (.status // ""),
  deterministic_group: (.deterministic_group // ""),
  output_hash: (.output_hash // "")
})) as $rows
| ($rows | length) as $n_rows
| ($rows
  | sort_by(.attn_requested, .cache_k_requested, .cache_v_requested, .ctx_len)
  | group_by([.attn_requested, .cache_k_requested, .cache_v_requested, .ctx_len])
  | map(cell_stats(.))
  | sort_by(.attn, .cache_k, .cache_v, .ctx_len)
 ) as $cells
| ($cells | map({ (key(.attn; .cache_k; .cache_v; .ctx_len)): . }) | add) as $cell_index

| ($rows | map(select(.attn_requested == "hisa" and .attn_observed == "hisa")) | length) as $hisa_observed_ok
| ($rows | map(select(.cache_k_requested == "q8_0" and .cache_v_requested == "turbo4" and .cache_k_observed == "q8_0" and .cache_v_observed == "turbo4")) | length) as $turbo_observed_ok
| ($rows | map(select(.attn_requested != .attn_observed or .cache_k_requested != .cache_k_observed or .cache_v_requested != .cache_v_observed)) | length) as $total_mismatch

| ($rows
  | map(select((.deterministic_group | length) > 0 and (.output_hash | length) > 0))
  | group_by(.deterministic_group)
  | map({group: .[0].deterministic_group, uniq_hashes: (map(.output_hash) | unique | length), runs: length})
 ) as $det_groups
| ($det_groups | map(select(.uniq_hashes > 1)) | length) as $det_fail_groups

| ([ $cells[]
    | select(.cache_k == "q8_0" and .cache_v == "turbo4")
    | . as $t
    | ($cell_index[key($t.attn; "f16"; "f16"; $t.ctx_len)] // null) as $b
    | select($b != null)
    | {
        attn: $t.attn,
        ctx_len: $t.ctx_len,
        mem_reduction_pct: (100 * (1 - ($t.kv_mem_mib_median / $b.kv_mem_mib_median))),
        gen_regression_pct: (100 * (1 - ($t.gen_tps_median / $b.gen_tps_median)))
      }
  ]) as $turbo_cmp
| ($turbo_cmp | map(select(.mem_reduction_pct >= $mem_min and .gen_regression_pct <= $gen_reg_max)) | length) as $turbo_cmp_pass

| ([ $cells[]
    | select(.attn == "hisa" and .cache_k == "f16" and .cache_v == "f16" and .ctx_len >= $hisa_min_ctx)
    | . as $h
    | ($cell_index[key("dense"; "f16"; "f16"; $h.ctx_len)] // null) as $d
    | select($d != null)
    | {
        ctx_len: $h.ctx_len,
        prefill_gain_pct: (100 * (($h.prompt_tps_median / $d.prompt_tps_median) - 1))
      }
  ]) as $hisa_cmp
| ($hisa_cmp | map(select(.prefill_gain_pct >= $hisa_gain_min)) | length) as $hisa_cmp_pass

| "# HISA/TurboQuant Validation Summary\n"
  + "Rows analyzed: \($n_rows)\n"
  + "\n## Cell Metrics (median/p95)\n"
  + "| attn | cache_k | cache_v | ctx | runs | prompt_tps_med | gen_tps_med | prompt_ms_p95 | gen_ms_p95 | kv_mem_mib_med | mismatches | status_fail |\n"
  + "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
  + ($cells
      | map("| \(.attn) | \(.cache_k) | \(.cache_v) | \(.ctx_len) | \(.runs) | \(fmt(.prompt_tps_median)) | \(fmt(.gen_tps_median)) | \(fmt(.prompt_ms_p95)) | \(fmt(.gen_ms_p95)) | \(fmt(.kv_mem_mib_median)) | \(.mismatch_count) | \(.status_fail_count) |")
      | join("\n"))
  + "\n\n## Checks\n"
  + "- Activation mismatch count: \($total_mismatch) => " + (if $total_mismatch == 0 then "PASS" else "FAIL" end) + "\n"
  + "- HISA observed runs: \($hisa_observed_ok) => " + (if $hisa_observed_ok > 0 then "PASS" else "INCONCLUSIVE" end) + "\n"
  + "- Turbo observed runs: \($turbo_observed_ok) => " + (if $turbo_observed_ok > 0 then "PASS" else "INCONCLUSIVE" end) + "\n"
  + "- Deterministic groups with hash drift: \($det_fail_groups) => " + (if $det_fail_groups == 0 then "PASS" else "FAIL" end) + "\n"
  + "- Turbo compare pass cells: \($turbo_cmp_pass)/\($turbo_cmp|length) (mem >= \($mem_min)% and gen regression <= \($gen_reg_max)%) => "
      + (if ($turbo_cmp|length) == 0 then "INCONCLUSIVE" elif $turbo_cmp_pass == ($turbo_cmp|length) then "PASS" else "FAIL" end) + "\n"
  + "- HISA prefill gain pass contexts: \($hisa_cmp_pass)/\($hisa_cmp|length) (ctx >= \($hisa_min_ctx), gain >= \($hisa_gain_min)%) => "
      + (if ($hisa_cmp|length) == 0 then "INCONCLUSIVE" elif $hisa_cmp_pass == ($hisa_cmp|length) then "PASS" else "FAIL" end) + "\n"

  + "\n## Turbo Compare Detail\n"
  + "| attn | ctx | mem_reduction_pct | gen_regression_pct |\n"
  + "|---|---:|---:|---:|\n"
  + (if ($turbo_cmp|length) == 0 then "| na | na | na | na |"
     else ($turbo_cmp
      | sort_by(.attn, .ctx_len)
      | map("| \(.attn) | \(.ctx_len) | \(pct(.mem_reduction_pct)) | \(pct(.gen_regression_pct)) |")
      | join("\n")) end)

  + "\n\n## HISA Prefill Gain Detail\n"
  + "| ctx | prefill_gain_pct (hisa f16/f16 vs dense f16/f16) |\n"
  + "|---:|---:|\n"
  + (if ($hisa_cmp|length) == 0 then "| na | na |"
     else ($hisa_cmp
      | sort_by(.ctx_len)
      | map("| \(.ctx_len) | \(pct(.prefill_gain_pct)) |")
      | join("\n")) end)
'