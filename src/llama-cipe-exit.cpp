#include "llama-cipe-exit.h"
#include "ggml.h"

#include <cmath>
#include <algorithm>
#include <numeric>

// Softmax function for converting logits to probabilities
void cipe_exit_softmax(std::vector<float> & logits) {
    if (logits.empty()) return;
    
    // Find maximum for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (float & logit : logits) {
        logit = std::exp(logit - max_logit);
        sum += logit;
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (float & logit : logits) {
            logit /= sum;
        }
    }
}

// Compute logits from hidden state using lm_head matrix
std::vector<float> cipe_exit_compute_logits(
    const ggml_tensor * hidden_state,
    const ggml_tensor * lm_head,
    ggml_context * ctx_compute
) {
    if (!hidden_state || !lm_head || !ctx_compute) {
        return {};
    }
    
    // Create computation graph for logits projection
    // ggml_tensor * logits = ggml_proj_to_logits(ctx_compute,
    //     const_cast<ggml_tensor*>(hidden_state),
    //     const_cast<ggml_tensor*>(lm_head));

    // For now, return empty vector as we need a way to evaluate the tensor
    // This would need to be integrated with the actual inference backend
    // In a real implementation, this would evaluate the tensor and return the results
    
    const int64_t n_vocab = lm_head->ne[1];
    std::vector<float> result(n_vocab, 0.0f);
    
    // TODO: Implement actual tensor evaluation
    // This is a placeholder that would need backend-specific evaluation
    
    return result;
}

// Compute KL divergence between two probability distributions
float cipe_exit_compute_kl_divergence(
    const std::vector<float> & logits1,
    const std::vector<float> & logits2
) {
    if (logits1.size() != logits2.size() || logits1.empty()) {
        return INFINITY;
    }
    
    // Convert logits to probabilities
    std::vector<float> p1 = logits1;
    std::vector<float> p2 = logits2;
    
    cipe_exit_softmax(p1);
    cipe_exit_softmax(p2);
    
    // Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    float kl_div = 0.0f;
    const float epsilon = 1e-10f; // Small value to avoid log(0)
    
    for (size_t i = 0; i < p1.size(); ++i) {
        if (p1[i] > epsilon && p2[i] > epsilon) {
            kl_div += p1[i] * std::log(p1[i] / p2[i]);
        }
    }
    
    return kl_div;
}

// Check if early exit should be triggered for current layer
bool cipe_exit_should_exit(
    llama_cipe_exit_context & ctx,
    const ggml_tensor * hidden_state,
    const ggml_tensor * lm_head,
    ggml_context * ctx_compute,
    int current_layer
) {
    if (!ctx.enabled || current_layer < ctx.min_layers) {
        return false;
    }
    
    // Compute current logits
    std::vector<float> current_logits = cipe_exit_compute_logits(
        hidden_state, lm_head, ctx_compute);
    
    if (current_logits.empty()) {
        return false;
    }
    
    // If we have previous logits, compute KL divergence
    if (!ctx.prev_logits.empty() && ctx.prev_logits.size() == current_logits.size()) {
        float kl_div = cipe_exit_compute_kl_divergence(ctx.prev_logits, current_logits);
        ctx.last_kl_divergence = kl_div;
        
        // Check if KL divergence is below threshold
        if (kl_div < ctx.threshold) {
            ctx.early_exit_triggered = true;
            return true;
        }
    }
    
    // Store current logits for next comparison
    ctx.prev_logits = current_logits;
    ctx.current_layer = current_layer;
    
    return false;
}

// Reset CIPE-Exit context for new sequence
void cipe_exit_reset(llama_cipe_exit_context & ctx) {
    ctx.current_layer = 0;
    ctx.early_exit_triggered = false;
    ctx.prev_logits.clear();
    ctx.last_kl_divergence = 0.0f;
    ctx.sequence_position = 0; // Reset sequence position for new prompt
}

// Get effective layer count based on CIPE-Exit optimization
int cipe_exit_get_effective_layers(
    const llama_cipe_exit_context & ctx,
    int total_layers
) {
    if (!ctx.enabled) {
        return total_layers;
    }

    int base_layers = ctx.adaptive_layer_count;

    // Sequence-aware adjustment: Use more layers at the beginning of sequences
    if (ctx.sequence_position < 5) {
        // First few tokens need more layers for better context understanding
        base_layers = std::min(total_layers, base_layers + 2);
    } else if (ctx.sequence_position > 20) {
        // Later tokens can use fewer layers (but not too few)
        base_layers = std::max(ctx.min_layers + 2, base_layers - 1);
    }

    return std::min(total_layers, std::max(ctx.min_layers, base_layers));
}

// Update CIPE-Exit context with KL divergence measurement from inference
void cipe_exit_update_kl_measurement(
    llama_cipe_exit_context & ctx,
    int layer,
    float kl_divergence
) {
    if (!ctx.enabled || layer < ctx.min_layers) {
        return;
    }

    ctx.current_layer = layer;
    ctx.last_kl_divergence = kl_divergence;

    // Check if early exit should be triggered
    if (kl_divergence < ctx.threshold) {
        ctx.early_exit_triggered = true;
    }
}

// Compute KL divergence from actual tensor data
float cipe_exit_compute_kl_from_tensors(
    const float * logits1,
    const float * logits2,
    int n_vocab
) {
    if (!logits1 || !logits2 || n_vocab <= 0) {
        return INFINITY;
    }

    // Convert logits to probabilities using softmax
    std::vector<float> p1(n_vocab), p2(n_vocab);

    // Find max for numerical stability
    float max1 = *std::max_element(logits1, logits1 + n_vocab);
    float max2 = *std::max_element(logits2, logits2 + n_vocab);

    // Compute softmax for both distributions
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        p1[i] = std::exp(logits1[i] - max1);
        p2[i] = std::exp(logits2[i] - max2);
        sum1 += p1[i];
        sum2 += p2[i];
    }

    // Normalize
    for (int i = 0; i < n_vocab; ++i) {
        p1[i] /= sum1;
        p2[i] /= sum2;
    }

    // Compute KL divergence: KL(P1||P2) = sum(P1 * log(P1/P2))
    float kl_div = 0.0f;
    const float epsilon = 1e-10f;

    for (int i = 0; i < n_vocab; ++i) {
        if (p1[i] > epsilon && p2[i] > epsilon) {
            kl_div += p1[i] * std::log(p1[i] / p2[i]);
        }
    }

    return kl_div;
}

// Update adaptive layer count based on inference results
void cipe_exit_update_adaptive_count(
    llama_cipe_exit_context & ctx,
    int total_layers,
    bool early_exit_occurred,
    int exit_layer
) {
    if (!ctx.enabled) {
        return;
    }

    ctx.total_inferences++;

    // Debug: printf("CIPE-Exit: Update adaptive count - inference #%d, early_exit=%s, exit_layer=%d, adaptive_count=%d\n",
    //        ctx.total_inferences, early_exit_occurred ? "true" : "false", exit_layer, ctx.adaptive_layer_count);

    if (early_exit_occurred && exit_layer >= ctx.min_layers) {
        ctx.successful_early_exits++;

        // Update average exit layer using exponential moving average
        const float alpha = 0.1f; // Learning rate
        ctx.avg_exit_layer = alpha * exit_layer + (1.0f - alpha) * ctx.avg_exit_layer;

        // Adjust adaptive layer count based on success rate
        const float success_rate = (float)ctx.successful_early_exits / ctx.total_inferences;

        // More conservative adaptive adjustment
        if (success_rate > 0.8f && ctx.total_inferences > 20) { // Very high success rate and enough data
            // Gradually reduce adaptive layer count, but not too aggressively
            const int target_layers = (int)(ctx.avg_exit_layer * 1.3f); // Add 30% buffer for safety
            if (target_layers > ctx.min_layers && target_layers < ctx.adaptive_layer_count) {
                ctx.adaptive_layer_count = std::max(ctx.min_layers, target_layers);
                // printf("CIPE-Exit: Reducing adaptive layers to %d (success_rate=%.2f, avg_exit=%.1f)\n",
                //        ctx.adaptive_layer_count, success_rate, ctx.avg_exit_layer);
            }
        } else if (success_rate < 0.3f && ctx.total_inferences > 10) { // Low success rate
            // Increase adaptive layer count for better quality
            int new_count = std::min(total_layers, ctx.adaptive_layer_count + 2); // Increase by 2 layers
            if (new_count != ctx.adaptive_layer_count) {
                ctx.adaptive_layer_count = new_count;
                // printf("CIPE-Exit: Increasing adaptive layers to %d (success_rate=%.2f) - improving quality\n",
                //        ctx.adaptive_layer_count, success_rate);
            }
        }
    }
}

// Check if output quality is degraded (simple heuristic)
bool cipe_exit_is_quality_degraded(
    const llama_cipe_exit_context & ctx
) {
    // If we're using very few layers and success rate is too high,
    // it might indicate quality degradation
    if (ctx.total_inferences > 50) {
        const float success_rate = (float)ctx.successful_early_exits / ctx.total_inferences;
        const float layer_ratio = (float)ctx.adaptive_layer_count / 26.0f; // Assuming 26 total layers

        // If using less than 60% layers with >95% success rate, quality might be degraded
        if (layer_ratio < 0.6f && success_rate > 0.95f) {
            return true;
        }
    }

    return false;
}

// Calculate real KL divergence between two probability distributions
float calculate_kl_divergence(const float* p, const float* q, const int size) {
    double kl_sum = 0.0;
    const double epsilon = 1e-9; // Prevent division by zero and log(0) errors

    for (int i = 0; i < size; ++i) {
        // Ensure input probabilities are positive
        if (p[i] > epsilon && q[i] > epsilon) {
            kl_sum += (double)p[i] * log((double)p[i] / (double)q[i]);
        }
    }

    return (float)kl_sum;
}

// Simple softmax function to convert logits to probabilities
void apply_softmax(const float* logits, float* probs, const int size) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; ++i) {
        probs[i] /= (float)sum;
    }
}
