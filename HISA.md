## Plan: Native HISA Runtime

Implement HISA as a native llama.cpp prefill attention mode, not as a new model architecture or GGUF conversion path. The recommended approach is backend-authoritative: add a runtime configuration surface, route prefill self-attention through a dedicated sparse-attention path, keep decode on dense/flash attention, and deliver full hisa-pytorch parity by splitting the work into runtime support plus profiling/calibration tooling.

**Steps**
1. Define the integration boundary and public control surface. Add a new prefill attention mode and HISA parameter group in the runtime API instead of overloading model architecture support. Keep flash attention as the dense-path optimizer and let HISA govern only long-context causal prefill. This blocks all later implementation work.
2. Add runtime configuration plumbing through shared params and CLI parsing. Mirror the new API fields into common params, CLI/env handling, and server/context construction so the feature can be enabled consistently across cli, server, and benches. *Depends on 1.*
3. Introduce HISA runtime state in the context/KV path. Add per-layer previous block selections, enabled-layer masks, per-layer top_m budgets, and reset hooks at prefill/sequence boundaries so cross-layer reuse has a principled place to live. *Depends on 1 and 2.*
4. Add a backend-facing sparse prefill attention abstraction in GGML. Implement a dedicated sparse prefill op or an equivalently isolated backend path with a CPU reference implementation for correctness and a CUDA implementation for performance. Do not model HISA as a new architecture and do not rely on ad hoc host-side string-parsed control flow. *Depends on 1. Can start in parallel with 3 once the parameter shape is settled.*
5. Wire HISA into the central attention builder. Update the shared attention construction path so causal self-attention during prefill can select HISA when enabled and supported, while generation, cross-attention, embeddings, and unsupported layouts automatically stay on dense/flash attention. *Depends on 3 and 4.*
6. Implement core HISA semantics inside the runtime path: block mean-pooling, coarse block scoring, forced boundary retention, local window retention, token index materialization, sparse K/V gather, and sparse causal attention. Start with correctness-first behavior, then add the CUDA-optimized path. *Depends on 5.*
7. Add full-parity cross-layer reuse. Use previous-layer selected blocks as runtime state feeding the next layer’s selection step, with explicit resets across requests and batch boundaries. Validate that reuse applies only within a single prefill pass and never leaks across sequences. *Depends on 6.*
8. Add adaptive per-layer profiling as tooling plus runtime input. Build a calibration flow that runs on representative prompts, computes per-layer top_m budgets, and saves/loads them for runtime use. This preserves full parity with hisa-pytorch without forcing heavyweight profiling logic into core model-loading APIs. *Depends on 2. Can run in parallel with 4 to 7 once the config format is defined.*
9. Add observability and fallback behavior. Log HISA enablement, backend support, fallback reasons, active budgets, and sparse-vs-dense call counts so benchmarking and debugging are practical. *Parallel with 5 to 8.*
10. Validate correctness and performance in stages. First prove dense fallback and disabled mode remain unchanged, then compare long-context logits/perplexity against dense baselines, and finally benchmark CUDA prefill latency and memory use at 4K, 8K, 16K, and 32K. *Depends on 5 to 9.*

**Relevant files**
- `/Users/davidroth/development/projects/llama-cpp-turboquant/include/llama.h` — add the public runtime API surface for a prefill attention mode and HISA parameter group; preserve the existing flash-attn API as a separate concern.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/common/common.h` — mirror new runtime settings into shared example/server params.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/common/arg.cpp` — add CLI/env parsing alongside the existing flash-attn option pattern.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/common/common.cpp` — copy shared params into context params.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/llama-cparams.h` — extend internal execution params used by graph construction.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/llama-context.cpp` — validate the new mode, decide auto/fallback behavior, own HISA runtime state resets, and log enablement/support decisions.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/llama-context.h` — hold any context-side state needed for per-request or per-layer HISA bookkeeping.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/llama-kv-cache.cpp` — hook sequence/prefill reset points if HISA state is stored with KV lifecycle.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/llama-graph.cpp` — change `llm_graph_context::build_attn_mha` to choose dense/flash vs HISA sparse prefill.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/ggml/include/ggml.h` — declare any new GGML op or backend-visible sparse-attention entry point.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/ggml/src/ggml.c` — define graph node creation, shape validation, and op metadata.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/ggml/src/ggml-cpu/ops.cpp` — implement a CPU reference path for correctness and testing.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/ggml/src/ggml-cuda/fattn.cu` — primary reference point for how a CUDA-first attention kernel is integrated; HISA likely belongs in a sibling CUDA path rather than model-specific code.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/src/models/deepseek2.cpp` — reference only; shows that `GLM_DSA` currently reuses DeepSeek2 graph building and is not the correct template for native HISA execution.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/tools/perplexity/perplexity.cpp` — correctness/regression validation on long prompts.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/tools/llama-bench/llama-bench.cpp` — CUDA-first prefill benchmarking across long contexts.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/tools/server/server-context.cpp` — inherit the new shared runtime params in server mode.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/tools/cli/README.md` — document the new runtime toggle once arg plumbing lands.
- `/Users/davidroth/development/projects/llama-cpp-turboquant/tools/server/README.md` — document server-facing enablement and limits.

**Verification**
1. Run existing dense inference with HISA disabled and confirm identical behavior and no graph/build regressions in cli and server.
2. Add correctness checks on a GQA model using long prefill prompts, comparing logits or perplexity between dense mode and HISA mode with agreed tolerances.
3. Verify decode-path fallback: prefill may use HISA, but token-by-token generation must remain dense/flash and produce stable outputs.
4. Exercise reset behavior with multiple requests/sequences to confirm cross-layer reuse state does not leak across prefills.
5. Benchmark CUDA prompt processing at 4K, 8K, 16K, and 32K and compare latency/VRAM to dense/flash baselines.
6. Validate calibration save/load by generating per-layer budgets once and reusing them across runs with reproducible results.
7. Check unsupported cases explicitly: embeddings, cross-attention, non-causal attention, and backends without HISA support should log a reason and fall back cleanly.

**Decisions**
- Included scope: native llama.cpp runtime feature, CUDA-first, full hisa-pytorch parity as the end state.
- Architectural decision: treat HISA as an attention execution mode, not as a new model architecture, new GGUF schema, or HF conversion feature.
- Runtime decision: apply HISA to long-context causal prefill only; generation remains dense/flash.
- Profiling decision: implement adaptive per-layer profiling as tooling that feeds runtime budgets, rather than baking a heavyweight profiler into core inference APIs on day one.
- Backend decision: prefer a backend-backed sparse prefill op/path over composing many generic graph ops if the goal is real CUDA speedup and maintainable backend authority.
- Discovery note: `LLM_ARCH_GLM_DSA` already carries sparse-attention-related metadata/tensors, but the current in-tree builder path mainly reuses `llm_build_deepseek2`; that should be treated as schema precedent, not as the native HISA execution template.

**Further Considerations**
1. Recommended staging: land the public/runtime scaffolding and a correctness-first reference path before committing to the optimized CUDA kernel details, but keep the API shape stable from the start so later work does not churn the surface area.
2. Recommended API shape: add a new prefill-attention mode plus a structured HISA parameter bundle rather than overloading `flash_attn_type`, because dense-kernel selection and sparse-prefill strategy are separate concerns.
3. Recommended parity interpretation: full parity should mean runtime support for cross-layer reuse and externally generated adaptive budgets, not necessarily a one-call profiler in the core C API in the first landing.