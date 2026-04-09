// TriAttention: Self-Calibrating Trigonometric KV Cache Token Eviction
//
// Inspired by "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression"
// by Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen
// arXiv:2604.04921, 2026
//
// This is an independent implementation that self-calibrates from Q projections
// during prefill — no external calibration files or Python dependencies needed.

#pragma once

#include "ggml.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class llama_kv_cache;

// Main TriAttention engine — self-calibrating scoring + eviction
struct llama_triattention {
    // Config
    int32_t budget        = 2048;
    int32_t divide_length = 128;
    int32_t window_size   = 128;

    // Model info
    uint32_t n_layers     = 0;
    uint32_t n_heads      = 0; // attention heads (Q heads)
    uint32_t n_kv_heads   = 0;
    uint32_t head_dim     = 0;
    uint32_t n_rot        = 0; // number of rotated dims (<= head_dim). For partial RoPE.
    uint32_t freq_count   = 0; // = n_rot / 2
    float    rope_theta   = 0.0f;

    // Precomputed RoPE angular frequencies
    std::vector<float> omega; // [freq_count]

    // Geometric offsets for score aggregation
    std::vector<float> offsets; // [1, 2, 4, ..., 65536]

    // Self-calibrated Q frequency statistics (the real deal)
    // Indexed as [layer * n_kv_heads + kv_head][freq_count]
    std::vector<std::vector<float>> center_real;
    std::vector<std::vector<float>> center_imag;
    std::vector<std::vector<float>> center_abs;

    // Accumulators for online Q stats during prefill
    // Same indexing as center_*
    std::vector<std::vector<float>> q_sum_real;
    std::vector<std::vector<float>> q_sum_imag;
    std::vector<std::vector<float>> q_sum_abs;
    int32_t q_samples = 0; // total tokens accumulated

    // Previous sums snapshot for EMA delta computation
    std::vector<std::vector<float>> q_prev_sum_real;
    std::vector<std::vector<float>> q_prev_sum_imag;
    std::vector<std::vector<float>> q_prev_sum_abs;
    int32_t q_samples_at_last_update = 0;

    // State
    bool calibrated = false;
    int32_t warmup_tokens          = 1024; // accumulate this many Q samples before first eviction
    float   ema_alpha              = 0.1f; // EMA blending factor (0.1 = stable, 0.5 = reactive)
    int32_t first_attn_layer       = -1;   // smallest attention-layer index ever observed (used to count tokens once per pass)

    // Initialize RoPE constants
    // n_rot: number of rotated dimensions (<= head_dim). Defaults to head_dim (full RoPE).
    void init_constants(float rope_theta, uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads, uint32_t n_layers, uint32_t n_rot = 0);

    // Accumulate Q statistics from a pre-RoPE Q tensor
    // stride_elems: stride in floats between consecutive heads (for 3D non-contig views).
    //   If 0, uses head_dim (standard contiguous layout).
    void accumulate_q(const float * q_data, int32_t n_tokens, int32_t layer_idx, size_t stride_elems = 0);

    // Update calibration: first call initializes, subsequent calls do EMA blend
    void update_calibration();

    // cb_eval callback — called during graph evaluation to capture pre-RoPE Q
    // Returns true if the tensor was consumed
    static bool eval_callback(ggml_tensor * t, bool ask, void * user_data);

    // Check whether eviction should be triggered
    bool should_evict(int32_t n_used) const;

    // Score tokens for one KV head in one layer
    void score_tokens(
        const float * k_data_f32,
        int32_t       n_tokens,
        int32_t       current_pos,
        int32_t       layer_idx,
        int32_t       kv_head_idx,
        float       * scores_out) const;

    // Run the full eviction pipeline
    int32_t evict(llama_kv_cache * cache, ggml_type type_k);
};
