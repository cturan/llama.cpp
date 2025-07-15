#pragma once

#include "llama-cparams.h"
#include "ggml.h"

#include <vector>

// CIPE-Exit utility functions for layer skipping optimization

// Compute logits from hidden state using lm_head matrix
std::vector<float> cipe_exit_compute_logits(
    const ggml_tensor * hidden_state,
    const ggml_tensor * lm_head,
    ggml_context * ctx_compute
);

// Compute KL divergence between two probability distributions
float cipe_exit_compute_kl_divergence(
    const std::vector<float> & logits1,
    const std::vector<float> & logits2
);

// Check if early exit should be triggered for current layer
bool cipe_exit_should_exit(
    llama_cipe_exit_context & ctx,
    const ggml_tensor * hidden_state,
    const ggml_tensor * lm_head,
    ggml_context * ctx_compute,
    int current_layer
);

// Reset CIPE-Exit context for new sequence
void cipe_exit_reset(llama_cipe_exit_context & ctx);

// Softmax function for converting logits to probabilities
void cipe_exit_softmax(std::vector<float> & logits);

// Get effective layer count based on CIPE-Exit optimization
// This can be used to dynamically reduce the number of layers processed
int cipe_exit_get_effective_layers(
    const llama_cipe_exit_context & ctx,
    int total_layers
);

// Update CIPE-Exit context with KL divergence measurement from inference
void cipe_exit_update_kl_measurement(
    llama_cipe_exit_context & ctx,
    int layer,
    float kl_divergence
);

// Process layers in a range [layer_start, layer_end) with CIPE-Exit optimization
// Returns the actual number of layers processed (may be less due to early exit)
int cipe_exit_process_layers_range(
    struct llama_context * ctx,
    const struct llama_batch & batch,
    int layer_start,
    int layer_end,
    bool & early_exit_triggered
);

// Compute KL divergence from actual tensor data
float cipe_exit_compute_kl_from_tensors(
    const float * logits1,
    const float * logits2,
    int n_vocab
);

// Update adaptive layer count based on inference results
void cipe_exit_update_adaptive_count(
    llama_cipe_exit_context & ctx,
    int total_layers,
    bool early_exit_occurred,
    int exit_layer
);

// Check if output quality is degraded (simple heuristic)
bool cipe_exit_is_quality_degraded(
    const llama_cipe_exit_context & ctx
);

// Phase 1: Observer Mode - Monitor KL divergence across layers
void cipe_exit_observe_kl_divergence(
    struct llama_context * ctx,
    const struct llama_batch & batch
);

// Calculate real KL divergence between two probability distributions
float calculate_kl_divergence(const float* p, const float* q, const int size);

// Simple softmax function to convert logits to probabilities
void apply_softmax(const float* logits, float* probs, const int size);
