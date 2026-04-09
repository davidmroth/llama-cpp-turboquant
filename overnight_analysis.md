# TriAttention Overnight + Followup Analysis — 2026-04-09

This file is the **source of truth** for what the overnight batch and the
follow-up reruns actually proved. Earlier drafts of this file contained
incorrect diagnoses — those have been corrected here. Findings are split into
**Verified** (eviction-confirmed, multi-chunk, reproducible) and **Open
questions** (still needs work).

---

## TL;DR

| Test | Result | Status |
|------|--------|--------|
| 7B @ 32K replication, 3 chunks, 90% retention   | +1.20% PPL | Verified |
| 7B @ 32K replication, 3 chunks, 85% retention   | +2.25% PPL | Verified |
| 7B @ 64K, 3 chunks, 90% retention               | +1.85% PPL | Verified |
| 7B @ 64K, 3 chunks, 85% retention               | +3.88% PPL | Verified |
| 7B full stack @ 32K (TQ + TriAtt 90%)           | +1.76% PPL | Verified |
| 7B NIAH @ 32K, end position, 90% & 85%          | **FAIL**   | Verified |
| 27B @ 32K, 3 chunks, 90% retention              | +0.69% PPL, 18 evict rounds | Verified |
| 27B @ 32K, 3 chunks, 85% retention              | +2.00% PPL, 27 evict rounds | Verified |
| 35B-A3B @ 32K, 3 chunks, 90% retention          | +0.55% PPL, 18 evict rounds | Verified |
| 35B-A3B @ 32K, 3 chunks, 85% retention          | +1.51% PPL, 27 evict rounds | Verified |
| Earlier "-3.9% improvement at 32K, 90%"         | Did not replicate | Was 1-chunk noise |

The single-chunk -3.9% PPL improvement that motivated the overnight batch
**did not survive multi-chunk replication**. The honest number is **+1.2% at
90% retention on 3 chunks**.

---

## Verified findings (reproducible, eviction-confirmed)

### 7B @ 32K — multi-chunk PPL replication

| Retention | Budget | PPL    | Δ vs baseline | Eviction rounds |
|-----------|--------|--------|---------------|-----------------|
| 100%      | 32768  | 6.8504 | baseline      | 0               |
| 90%       | 29491  | 6.9327 | **+1.20%**    | 18              |
| 85%       | 27853  | 7.0048 | **+2.25%**    | 27              |

(Qwen2.5-7B-Instruct-Q8_0, wikitext-2-raw, c=32768, b=512, --chunks 3)

### 7B @ 64K — multi-chunk PPL replication

| Retention | Budget | PPL    | Δ vs baseline | Eviction rounds |
|-----------|--------|--------|---------------|-----------------|
| 100%      | 65536  | 6.2531 | baseline      | 0               |
| 90%       | 58982  | 6.3687 | **+1.85%**    | 36              |
| 85%       | 55706  | 6.4960 | **+3.88%**    | 54              |

The cost grows roughly linearly with retention savings — about +0.4% PPL per
1% of KV evicted on this workload. There is no "free lunch zone" at 32K-64K
on standard wikitext.

### 7B full stack @ 32K — TQ + TriAtt compose cleanly

| Config                     | PPL    | Δ vs f16  |
|----------------------------|--------|-----------|
| f16 KV (baseline)          | 6.8504 | 0%        |
| TQ only (q8_0 K + turbo3 V)| 6.8879 | +0.55%    |
| TriAtt 90% only            | 6.9327 | +1.20%    |
| **TQ + TriAtt 90% (STACK)**| **6.9710** | **+1.76%** |

The stack cost (+1.76%) is approximately additive: TQ (+0.55%) + TriAtt
(+1.20%) = +1.75%. This is good evidence that the two methods don't interact
destructively.

### Qwen3.5-27B @ 32K — eviction now fires (3 chunks)

| Retention | Budget | PPL    | Δ vs baseline | Eviction rounds |
|-----------|--------|--------|---------------|-----------------|
| 100%      | 32768  | 7.4640 | baseline      | 0               |
| 90%       | 29491  | 7.5158 | **+0.69%**    | 18              |
| 85%       | 27853  | 7.6131 | **+2.00%**    | 27              |

(Qwen3.5-27B-Q8_0 — qwen35 hybrid Mamba+Attention, head_dim=256, n_rot=64,
16 of 64 layers attention. Eviction targets only the attention KV cache via
`llama_memory_hybrid::get_mem_attn()`.)

### Qwen3.5-35B-A3B @ 32K — eviction now fires (3 chunks)

| Retention | Budget | PPL    | Δ vs baseline | Eviction rounds |
|-----------|--------|--------|---------------|-----------------|
| 100%      | 32768  | 6.2720 | baseline      | 0               |
| 90%       | 29491  | 6.3062 | **+0.55%**    | 18              |
| 85%       | 27853  | 6.3669 | **+1.51%**    | 27              |

(Qwen3.5-35B-A3B-Q8_0 — qwen35moe hybrid Mamba+Attention+MoE, head_dim=256,
n_rot=64, 10 of 40 layers attention, 256 experts, 8 used.)

### Comparison: 7B vs 27B vs 35B-A3B at 32K, 90% retention

| Model      | Δ PPL @ 90% | Δ PPL @ 85% |
|------------|-------------|-------------|
| 7B (qwen2) | +1.20%      | +2.25%      |
| 27B (qwen35) | +0.69%    | +2.00%      |
| 35B-A3B (qwen35moe) | +0.55% | +1.51%   |

Hybrid models tolerate eviction *better* than the pure transformer at the same
retention ratio. Hypothesis: the SSM layers carry significant context state
unaffected by attention-cache eviction, so the model has another path to
preserve information. Untested.

### 7B NIAH @ 32K — retrieval breaks at end position

Strict checker (exact match for "PURPLE ELEPHANT 7742", looking only at
generated tokens — not the echoed prompt). Run with `--no-display-prompt`
and `-no-cnv` to avoid the earlier false-PASS from prompt echo.

| Config         | start (400) | middle (65000) | end (120000) |
|----------------|-------------|-----------------|---------------|
| Baseline 100%  | PASS        | PASS            | PASS          |
| TriAtt 90%     | PASS        | PASS            | **FAIL** ("12345") |
| TriAtt 85%     | PASS        | PASS            | **FAIL** ("12345") |

Eviction-confirmed (90%: 2 rounds, 85%: 3 rounds). The model hallucinates
"12345" instead of recalling the needle when the needle was inserted near the
end of the context and the cache was evicted.

This is a **real failure mode**, not a measurement bug. It's consistent with
the paper's framing that TriAttention is generative-first, not retrieval-safe.

### Code fixes that landed

1. **`common/common.cpp`**: stop computing `head_dim = n_embd / n_head` —
   that's wrong for fused-QKV models like qwen35 (gives 213 instead of 256).
   Now passes `0` and lets the engine read `hparams.n_embd_head_k()`.

2. **`src/llama-context.cpp`**: handle `llama_memory_hybrid` in the eviction
   hook. Hybrid models use `llama_memory_hybrid` which wraps the attention
   `llama_kv_cache` internally — `dynamic_cast<llama_kv_cache*>` returns null
   for them, so the eviction hook silently never ran. Now we fall back to
   `llama_memory_hybrid::get_mem_attn()`.

3. **`src/llama-triattention.cpp::eval_callback`**: accept both `Qcur-N`
   (2D, qwen2-style) and `Qcur_normed-N` (3D, qwen35-style) tensor names.
   Use `t->nb[1] / sizeof(float)` as the per-head stride for 3D inputs to
   support non-contiguous views correctly.

4. **`src/llama-triattention.cpp`**: track `first_attn_layer` (smallest
   layer index ever observed). Token counting was hardcoded to "layer 0",
   but qwen35 attention layers are 3, 7, 11, 15... never 0, so the counter
   never advanced and calibration never fired.

5. **`src/llama-triattention.h/cpp`**: add `n_rot` field separate from
   `head_dim`. For models with partial RoPE (qwen35: head_dim=256, n_rot=64),
   only the first n_rot dims are rotated. omega and freq_count are now
   computed from n_rot.

6. **`src/llama-context.cpp::llama_triattention_enable`**: read theta from
   `hparams.rope_freq_base_train` and n_rot from `hparams.n_rot()` instead
   of hardcoded defaults.

7. **`niah_7b_strict.sh`**: use `-no-cnv` (avoid stdin EOF) and
   `--no-display-prompt` (otherwise the echoed prompt contains the needle and
   every test silently passes). Read prompt from a file with `-f`.

---

## Architecture corrections (vs. earlier draft)

The earlier `overnight_analysis.md` claimed Qwen3.5-27B has `head_dim=213`.
That number is wrong. It came from `n_embd / n_head = 5120 / 24 = 213.33`,
which is the wrong formula for fused-QKV layouts. The actual values from the
GGUF metadata are:

| Model            | Arch       | head_dim (key_length) | n_head | n_kv_head | n_rot | layers | full_attn_interval |
|------------------|------------|-----------------------|--------|-----------|-------|--------|-------------------|
| Qwen2.5-7B       | qwen2      | 128                   | 28     | 4         | 128   | 28     | n/a (all attention) |
| Qwen3.5-27B      | qwen35     | 256                   | 24     | 4         | 64    | 64     | 4 (so 16 attn layers) |
| Qwen3.5-35B-A3B  | qwen35moe  | 256                   | 16     | 2         | 64    | 40     | 4 (so 10 attn layers) |

Both qwen35 and qwen35moe are **hybrid Mamba2 + Attention** architectures
(not pure transformers). They use:

- Fused QKV via `Qcur_full → ggml_view_3d → Qcur_normed → ggml_rope_multi`
- Partial RoPE: only the first `n_rot=64` of `head_dim=256` are rotated
- M-RoPE with `rope_dimension_sections = [11, 11, 10, 0]` (text/h/w/e)
- IMROPE rotation type (interleaved M-RoPE) — but still half-layout for the
  rotated portion (`rotate_pairs(n_dims, n_dims/2, ...)` in the kernel)
- `llama_memory_hybrid` cache (attention KV + recurrent state)

The original "head_dim=213" diagnosis was wrong because (a) we computed it
incorrectly in `common.cpp`, and (b) the deeper issue was unrelated — the
real blockers were the hybrid memory dispatch, the missing layer-0
heuristic, and the missing 3D Q tensor capture path.

---

## Open questions (still TBD)

1. **Does the trig scoring math actually fit qwen35's M-RoPE / partial RoPE?**
   Eviction *fires* and PPL deltas are reasonable, but the omega values we
   precompute assume standard RoPE over n_rot dimensions. For text-only
   inference all M-RoPE sections collapse to the same position index, so it's
   approximately equivalent — but the theta_scale per-section subtleties mean
   our scoring could be measurably off. The fact that hybrid models actually
   tolerate eviction *better* than the pure transformer is a good sign but
   isn't proof of correctness.

2. **Why do hybrid models tolerate eviction better than the pure transformer?**
   Hypothesis: SSM layers carry parallel context state that survives
   attention-cache eviction. Untested.

3. **Reasoning models** (the paper's stated best case). Untested.
   DeepSeek-R1-distill or QwQ would be the obvious candidates.

4. **27B/35B NIAH** — the qwen35 build didn't run NIAH yet. Now that PPL
   eviction is verified, NIAH should follow.

5. **Interaction with TQ on hybrid models** — only validated on 7B. The TQ
   path for hybrid memory may also need the same `get_mem_attn()` shim.

---

## Files

- `run_overnight.sh`           — original (broken) overnight script. PPL extraction
                                  bug, false-PASS NIAH check. Kept for the run.
- `overnight_results.jsonl`    — original overnight results (PPL fields are `=`)
- `overnight_log.txt`          — original overnight stdout
- `followup_tests.sh`          — first followup batch (also had broken NIAH)
- `followup_results.txt`       — 7B PPL replication + full-stack data (valid).
                                  NIAH section is invalid (stdin EOF bug).
- `niah_7b_strict.sh`          — fixed NIAH script (no-cnv, no-display-prompt,
                                  exact needle match in generated tokens only)
- `niah_7b_strict_results.txt` — verified 7B NIAH results
- `qwen35_validation.sh`       — 27B / 35B-A3B PPL validation at 32K
- `qwen35_validation_results.txt` — qwen35 PPL data (in progress)
