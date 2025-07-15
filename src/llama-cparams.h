#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// CIPE-Exit context for tracking layer skipping state
struct llama_cipe_exit_context {
    bool enabled;
    float threshold;
    int32_t min_layers;
    int32_t current_layer;
    bool early_exit_triggered;
    float last_kl_divergence;

    // Adaptive layer count based on historical performance
    int32_t adaptive_layer_count;    // Current adaptive layer count
    int32_t total_inferences;        // Total number of inferences processed
    int32_t successful_early_exits;  // Number of successful early exits
    float avg_exit_layer;            // Average layer where early exit occurred
    int32_t sequence_position;       // Position in current sequence (for dynamic adjustment)

    // Real KL divergence calculation storage
    std::vector<float> prev_logits;  // Previous layer logits for KL divergence
    bool has_prev_logits;            // Whether we have previous logits to compare
};

#define LLAMA_MAX_SEQ 64

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int      n_threads;       // number of threads to use for generation
    int      n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    // CIPE-Exit parameters for layer skipping optimization
    bool    cipe_exit;
    float   cipe_exit_threshold;
    float   cipe_exit_start_thr;
    float   cipe_exit_end_thr;
    int32_t min_layers_to_run;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool no_perf;
    bool warmup;
    bool op_offload;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // CIPE-Exit context
    llama_cipe_exit_context cipe_exit_ctx;
};
