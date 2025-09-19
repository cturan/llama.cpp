#include "ggml.h"
#include "ggml-delta.h"
#include "ggml-impl.h"

static void report_tensor_size(const char * tensor_name, const struct ggml_tensor * tensor) {
    GGML_LOG_INFO("[%s] tensor size is [%lu, %lu, %lu, %lu], strides [%lu, %lu, %lu, %lu]\n", 
        tensor_name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
}

// ggml_delta_net
struct ggml_tensor * ggml_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * q,
        struct ggml_tensor  * g,
        struct ggml_tensor  * conv_weight,
        struct ggml_tensor  * conv_bias,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 scale) {
    
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));
    report_tensor_size("orig_k", k);
    report_tensor_size("orig_v", v);
    report_tensor_size("orig_q", q);
    report_tensor_size("orig_g", g);
    report_tensor_size("orig_beta", beta);
    report_tensor_size("orig_state", state);
    
    const int64_t S_k = k->ne[0];
    const int64_t H_k = k->ne[1];
    const int64_t batch_size = k->ne[2];  
    const int64_t n_tokens = k->ne[3];
    
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    
    GGML_ASSERT(v->ne[3] == n_tokens);
    GGML_ASSERT(q->ne[3] == n_tokens);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[1] == batch_size && beta->ne[2] == n_tokens && beta->ne[3] == 1);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == 1);
    
    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[3] == n_tokens);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[3] == n_tokens);
       
    // Validate g dimensions - g should be [S_v, H_v, n_tokens, batch_size] based on actual tensor layout
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[3] == n_tokens && g->ne[2] == batch_size);
    
    // Apply sigmoid to beta for gating
    struct ggml_tensor * beta_sigmoid = ggml_sigmoid(ctx, beta);
    report_tensor_size("beta_sigmoid", beta_sigmoid);
    
    // Concatenate q, k, v for convolution processing
    struct ggml_tensor * mixed_qkv = ggml_concat(ctx, q, k, 1);
    report_tensor_size("mixed_qkv_qk", mixed_qkv);
    mixed_qkv = ggml_concat(ctx, mixed_qkv, v, 1);
    report_tensor_size("mixed_qkv_qkv", mixed_qkv);

    uint32_t dim = (S_v * H_v) + 2 * (H_k * S_k);

    mixed_qkv = ggml_reshape_3d(ctx, mixed_qkv, batch_size, dim, n_tokens);
    report_tensor_size("mixed_qkv_reshaped", mixed_qkv);
    struct ggml_tensor * mixed_qkv_padded = ggml_pad(ctx, mixed_qkv, conv_weight->ne[0] - 1, 0, 0, 0);
    report_tensor_size("mixed_qkv_padded", mixed_qkv_padded);

    // Apply SSM convolution
    struct ggml_tensor * conv_out = ggml_ssm_conv(ctx, mixed_qkv_padded, conv_weight);
    report_tensor_size("conv_out", conv_out);

    // Apply bias if provided
    if (conv_bias) {
        conv_out = ggml_add(ctx, conv_out, conv_bias);
        report_tensor_size("conv_out_bias", conv_out);
    }

    // Apply SiLU activation
    conv_out = ggml_silu(ctx, conv_out);
    report_tensor_size("conv_out_silu", conv_out);

    // Reshape back to 4D: [dim, n_tokens, 1] -> [dim, n_tokens, 1, 1]
    conv_out = ggml_reshape_4d(ctx, conv_out, dim, n_tokens, batch_size, 1);
    report_tensor_size("conv_out_reshaped", conv_out);

    // Transpose to get the right layout: [dim, n_tokens, 1] -> [dim, 1, n_tokens, 1]
    conv_out = ggml_permute(ctx, conv_out, 0, 2, 1, 3);
    report_tensor_size("conv_out_transposed", conv_out);

    // q projection view
    struct ggml_tensor * q_conv = ggml_view_4d(ctx, conv_out,
                                               S_k,                  // ne0
                                               H_k,                  // ne1
                                               conv_out->ne[1],      // ne2 = sequence length (1)
                                               conv_out->ne[2],      // ne3 = batch (1)
                                               H_k * sizeof(float),  // nb1 = stride along H_k
                                               conv_out->nb[1],      // nb2 = stride along sequence dim
                                               conv_out->nb[2],      // nb3 = stride along batch dim
                                               0                     // offset in bytes
    );
    report_tensor_size("q_conv_view", q_conv);

    // k projection view
    struct ggml_tensor * k_conv = ggml_view_4d(ctx, conv_out,
                                               S_k,                       // ne0
                                               H_k,                       // ne1
                                               conv_out->ne[1],           // ne2
                                               conv_out->ne[2],           // ne3
                                               H_k * sizeof(float),       // nb1
                                               conv_out->nb[1],           // nb2
                                               conv_out->nb[2],           // nb3
                                               S_k * H_k * sizeof(q->type)  // offset = skip q_out
    );
    report_tensor_size("k_conv_view", k_conv);

    // v projection view
    struct ggml_tensor * v_conv = ggml_view_4d(ctx, conv_out,
                                               S_v,                             // ne0
                                               H_v,                             // ne1
                                               conv_out->ne[1],                 // ne2
                                               conv_out->ne[2],                 // ne3
                                               H_v * sizeof(float),             // nb1
                                               conv_out->nb[1],                 // nb2
                                               conv_out->nb[2],                 // nb3
                                               (2 * S_k * H_k) * sizeof(q->type)// offset = skip q_out + k_out
    );
    report_tensor_size("v_conv_view", v_conv);

    // Transpose each component back to original layout: [S_v, 1, token_split_size, 1] -> [S_v, token_split_size, 1, 1]
    q_conv = ggml_permute(ctx, q_conv, 0, 2, 1, 3);
    report_tensor_size("q_conv_permuted", q_conv);
    k_conv = ggml_permute(ctx, k_conv, 0, 2, 1, 3);
    report_tensor_size("k_conv_permuted", k_conv);
    v_conv = ggml_permute(ctx, v_conv, 0, 2, 1, 3);
    report_tensor_size("v_conv_permuted", v_conv);

    q_conv = ggml_reshape_3d(ctx, ggml_cont(ctx, q_conv), S_k * H_k, batch_size, n_tokens);
    report_tensor_size("q_conv_reshaped", q_conv);
    k_conv = ggml_reshape_3d(ctx, ggml_cont(ctx, k_conv), S_k * H_k, batch_size, n_tokens);
    report_tensor_size("k_conv_reshaped", k_conv);
    v_conv = ggml_reshape_3d(ctx, ggml_cont(ctx, v_conv), S_v * H_v, batch_size, n_tokens);
    report_tensor_size("v_conv_reshaped", v_conv);
    
    // NOW we repeat query and key to match value head dimensions if needed (after convolution)
    struct ggml_tensor * q_broadcast = q_conv;
    struct ggml_tensor * k_broadcast = k_conv;
    
    if (H_k != H_v) {
        // Calculate the repeat factor: H_v / H_k
        GGML_ASSERT(H_v % H_k == 0);
        int64_t repeat_factor = H_v / H_k;
        
        // Repeat query and key along the head dimension
        // First reshape to separate the repeat dimension: [S_k, H_k, n_tokens, 1] -> [S_k, 1, H_k, n_tokens]
        q_broadcast = ggml_reshape_4d(ctx, q_conv, S_k, batch_size, H_k, n_tokens);
        report_tensor_size("q_broadcast_reshape1", q_broadcast);
        k_broadcast = ggml_reshape_4d(ctx, k_conv, S_k, batch_size, H_k, n_tokens);
        report_tensor_size("k_broadcast_reshape1", k_broadcast);
        
        // Repeat along the new dimension: [S_k, repeat_factor, H_k, n_tokens]
        q_broadcast = ggml_repeat_4d(ctx, q_broadcast, S_k, batch_size * repeat_factor, H_k, n_tokens);
        report_tensor_size("q_broadcast_repeat", q_broadcast);
        k_broadcast = ggml_repeat_4d(ctx, k_broadcast, S_k, batch_size * repeat_factor, H_k, n_tokens);
        report_tensor_size("k_broadcast_repeat", k_broadcast);
        
        // Reshape back to original dimensions but with H_v heads: [S_k, H_v, n_tokens, 1]
        q_broadcast = ggml_reshape_4d(ctx, q_broadcast, S_k, H_v, n_tokens, batch_size);
        report_tensor_size("q_broadcast_reshape2", q_broadcast);
        k_broadcast = ggml_reshape_4d(ctx, k_broadcast, S_k, H_v, n_tokens, batch_size);
        report_tensor_size("k_broadcast_reshape2", k_broadcast);
    }

    struct ggml_tensor * v_reshape = ggml_reshape_4d(ctx, v_conv, S_v, H_v, n_tokens, batch_size);
    report_tensor_size("v_reshape", v_reshape);
    struct ggml_tensor * v_broadcast = ggml_repeat_4d(ctx, v_reshape, S_v, H_v, n_tokens, batch_size);
    report_tensor_size("v_broadcast", v_broadcast);
    // g already has correct dimensions [S_v, H_v, n_tokens, batch_size], no need to reshape
    struct ggml_tensor * g_reshape = g;
    report_tensor_size("g_reshape", g_reshape);
    q_broadcast = ggml_repeat_4d(ctx, q_broadcast, S_k, H_v, n_tokens, batch_size);
    report_tensor_size("q_broadcast_final", q_broadcast);
    k_broadcast = ggml_repeat_4d(ctx, k_broadcast, S_k, H_v, n_tokens, batch_size);
    report_tensor_size("k_broadcast_final", k_broadcast);
    struct ggml_tensor * beta_reshape = ggml_reshape_4d(ctx, beta_sigmoid, 1, H_v, n_tokens, batch_size);
    report_tensor_size("beta_reshape", beta_reshape);
    struct ggml_tensor * beta_broadcast = ggml_repeat_4d(ctx, beta_reshape, 1, H_v, n_tokens, batch_size);
    report_tensor_size("beta_broadcast", beta_broadcast);
    // The state should be repeated along the sequence dimension only
    // Original state: [S_v, S_v, H_v, 1] -> should become [S_v, S_v, H_v, n_seqs]
    // Use ggml_cont to ensure the state is contiguous, not ggml_repeat_4d which would repeat along all dimensions
    struct ggml_tensor * state_broadcast = ggml_cont(ctx, state);
    report_tensor_size("state_broadcast", state_broadcast);
    
    // Call tensor-level kernel with convolved and processed tensors
    return ggml_delta_net_op(ctx, q_broadcast, k_broadcast, v_broadcast, g_reshape, beta_broadcast, state_broadcast, use_qk_l2norm, scale);
}

struct ggml_tensor * ggml_delta_net_op(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 scale) {
    
    // Debug: Log input tensor dimensions
    report_tensor_size("q_input", q);
    report_tensor_size("k_input", k);
    report_tensor_size("v_input", v);
    report_tensor_size("g_input", g);
    report_tensor_size("beta_input", beta);
    report_tensor_size("state_input", state);
    
    // Validate dimensions
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));
    
    const int64_t S_k = q->ne[0];  // head dimension for q/k
    const int64_t H_k = q->ne[1];  // number of heads (already processed to match v)
    const int64_t n_tokens = q->ne[2];
    const int64_t batch_size = q->ne[3];  // batch size, not n_seqs
    
    const int64_t S_v = v->ne[0];  // head dimension for v
    const int64_t H_v = v->ne[1];  // head dimension for v

    GGML_LOG_INFO("S_k = %ld, S_v = %ld, H_k = %ld, H_v = %ld\n", S_k, S_v, H_k, H_v);
    
    // Validate dimensions - match Python implementation layout
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_v && k->ne[2] == n_tokens && k->ne[3] == batch_size);
    GGML_ASSERT(v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == batch_size);
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[3] == n_tokens && g->ne[2] == batch_size);
    GGML_ASSERT(beta->ne[0] == 1 && beta->ne[1] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == batch_size);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_tokens);
    
    struct ggml_tensor * output = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v, H_v, batch_size, n_tokens);
    report_tensor_size("output", output);
    
    struct ggml_tensor * new_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v * S_v, H_v, 1, n_tokens);
    
    // Copy initial state to new_state
    new_state = ggml_cpy(ctx, state, new_state);
    report_tensor_size("new_state_copied", new_state);
    
    // Process all sequences and heads together using tensor operations
    
    // Apply L2 normalization if requested - per head, token, and sequence
    if (use_qk_l2norm) {
        q = ggml_l2_norm(ctx, q, 1e-6f);
        report_tensor_size("q_l2norm", q);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        report_tensor_size("k_l2norm", k);
    }
    
    // Apply scaling to query - across all tokens, sequences and heads
    q = ggml_scale(ctx, q, scale);
    report_tensor_size("q_scaled", q);
    
    // Process the gated delta rule using tensor operations
    
    // Reshape for matrix operations: [S_v, S_v, H_v, 1] -> [S_v * S_v, H_v]
    struct ggml_tensor * state_flat = ggml_reshape_2d(ctx, new_state, S_v * S_v, H_v);
    report_tensor_size("state_flat", state_flat);
    
    // Process each token sequentially due to recurrent nature
    for (int64_t t = 0; t < n_tokens; ++t) {
        // Extract current token's data across all batches and heads
        // q, k, v are [S_k, H_k, n_tokens, batch_size] layout in GGML
        struct ggml_tensor * q_t = ggml_view_3d(ctx, q, S_k, H_k, batch_size,
                                               q->nb[1], q->nb[2], t * q->nb[2]);
        report_tensor_size("q_t_view", q_t);
        struct ggml_tensor * k_t = ggml_view_3d(ctx, k, S_k, H_k, batch_size,
                                               k->nb[1], k->nb[2], t * k->nb[2]);
        report_tensor_size("k_t_view", k_t);
        struct ggml_tensor * v_t = ggml_view_3d(ctx, v, S_v, H_v, batch_size,
                                               v->nb[1], v->nb[2], t * v->nb[2]);
        report_tensor_size("v_t_view", v_t);
        struct ggml_tensor * beta_t = ggml_view_3d(ctx, beta, 1, H_v, batch_size,
                                                  beta->nb[1], beta->nb[2], t * beta->nb[2]);
        report_tensor_size("beta_t_view", beta_t);
                
        // Simplified approach: follow Python implementation exactly
        // In Python: kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        // This means: for each batch and head, multiply state by k_t and sum over the last dimension
        
        // First, reshape tensors to match GGML layout for head-wise processing
        // q_t: [S_k, H_k, batch_size] -> reshape to [S_k, H_k * batch_size]
        struct ggml_tensor * q_t_reshaped = ggml_reshape_2d(ctx, q_t, S_k, H_k * batch_size);
        report_tensor_size("q_t_reshaped", q_t_reshaped);
        
        // k_t: [S_k, H_k, batch_size] -> reshape to [S_k, H_k * batch_size]
        struct ggml_tensor * k_t_reshaped = ggml_reshape_2d(ctx, k_t, S_k, H_k * batch_size);
        report_tensor_size("k_t_reshaped", k_t_reshaped);
        
        // v_t: [S_v, H_v, batch_size] -> reshape to [S_v, H_v * batch_size]
        struct ggml_tensor * v_t_reshaped = ggml_reshape_2d(ctx, v_t, S_v, H_v * batch_size);
        report_tensor_size("v_t_reshaped", v_t_reshaped);
        
        // beta_t: [1, H_v, batch_size] -> reshape to [1, H_v * batch_size]
        struct ggml_tensor * beta_t_reshaped = ggml_reshape_2d(ctx, beta_t, 1, H_v * batch_size);
        report_tensor_size("beta_t_reshaped", beta_t_reshaped);
        
        // Handle head dimension mismatch - repeat k_t if needed
        struct ggml_tensor * k_t_final = k_t_reshaped;
        if (H_k != H_v) {
            GGML_ASSERT(H_v % H_k == 0);
            
            // Reshape k_t to separate head and batch dimensions: [S_k, H_k, batch_size, 1]
            struct ggml_tensor * k_t_4d = ggml_reshape_4d(ctx, k_t_reshaped, S_k, H_k, 1, batch_size);
            report_tensor_size("k_t_4d", k_t_4d);
            
            // Repeat along head dimension: [S_k, H_v, batch_size, 1]
            k_t_final = ggml_repeat_4d(ctx, k_t_4d, S_k, H_v, 1, batch_size);
            report_tensor_size("k_t_final_repeated", k_t_final);
            
            // Reshape back to 2D: [S_k, H_v * batch_size]
            k_t_final = ggml_reshape_2d(ctx, k_t_final, S_k, H_v * batch_size);
            report_tensor_size("k_t_final_2d", k_t_final);
        }
        
        // Simplified kv_mem computation: state @ k_t^T for each head
        // For now, let's use a simpler approach that matches the Python logic more closely
        // kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        
        // Reshape state to [S_v * S_v, H_v] for easier processing
        struct ggml_tensor * state_2d = ggml_reshape_2d(ctx, new_state, S_v * S_v, H_v);
        report_tensor_size("state_2d", state_2d);
        
        // The state is already in the correct format for matrix operations
        struct ggml_tensor * state_t = state_2d;
        report_tensor_size("state_t", state_t);
        
        // Simple kv_mem computation for this token
        // kv_mem = (state_t * k_t.unsqueeze(-1)).sum(dim=-2)
        // In GGML, we need to implement: (state_t * k_t_broadcast).sum(dim=1)
        // state_t: [S_v * S_v, H_v], k_t_final: [S_k, H_v * batch_size]
        
        // For the correct matrix multiplication, we need:
        // state_t: [S_v * S_v, H_v]
        // k_t_final: [S_k, H_v * batch_size]
        // We want: state_t @ k_t_transposed where k_t_transposed is [H_v * batch_size, S_k]
        
        // But first, let's check if we can do a simpler approach
        // Since we have H_v = 16 and batch_size = 1, we have:
        // state_t: [16384, 16] and k_t_final: [128, 16]
        
        // For matrix multiplication, we need: [16384, 16] @ [16, 128] = [16384, 128]
        // So we need to transpose k_t_final to get [16, 128]
        
        // For GGML matrix multiplication, we need to satisfy ggml_can_mul_mat requirements:
        // t0->ne[0] == t1->ne[0] (first dimensions must be equal)
        // t1->ne[2]%t0->ne[2] == 0 (broadcastable along 3rd dimension)
        // t1->ne[3]%t0->ne[3] == 0 (broadcastable along 4th dimension)
        
        // We need to reshape state_t from [S_v * S_v, H_v, 1, 1] to [H_v, S_v * S_v, 1, 1]
        // and k_t_final from [S_k, H_v * batch_size] to [H_v, S_k, 1, 1]
        
        // First, transpose state_t to get [H_v, S_v * S_v, 1, 1]
        struct ggml_tensor * state_t_transposed = ggml_transpose(ctx, state_t);
        report_tensor_size("state_t_transposed", state_t_transposed);
        
        // Reshape k_t_final from [S_k, H_v * batch_size] to [H_v, S_k, 1, 1]
        struct ggml_tensor * k_t_final_reshaped = ggml_reshape_4d(ctx, k_t_final, H_v, S_k, batch_size, 1);
        report_tensor_size("k_t_final_reshaped", k_t_final_reshaped);
        
        // Now we can do matrix multiplication: k_t_final_reshaped^T @ state_t_transposed^T
        // But GGML doesn't allow transposed first argument, so we need to swap the order
        // and transpose the result if needed
        struct ggml_tensor * kv_mem = ggml_mul_mat(ctx, k_t_final_reshaped, state_t_transposed);
        report_tensor_size("kv_mem", kv_mem);
                
        // Compute delta = (v_t - kv_mem) * beta_t
        // kv_mem: [batch_size, S_v] (result of state @ k_t^T)
        // v_t: [batch_size, H_v, S_v] -> reshape to [batch_size * H_v, S_v]
        // beta_t: [batch_size, H_v, 1] -> reshape to [batch_size * H_v, 1]
        
        // Handle head dimension mismatch for v_t and beta_t
        struct ggml_tensor * v_t_final = v_t_reshaped;
        struct ggml_tensor * beta_t_final = beta_t_reshaped;
        
        if (H_k != H_v) {
            // Repeat v_t and beta_t along head dimension to match H_v
            // v_t: [S_v, H_k, batch_size] -> [S_v, H_k, batch_size, 1] -> repeat -> [S_v, H_v, batch_size, 1]
            struct ggml_tensor * v_t_4d = ggml_reshape_4d(ctx, v_t_reshaped, S_v, H_k, 1, batch_size);
            struct ggml_tensor * v_t_repeated = ggml_repeat_4d(ctx, v_t_4d, S_v, H_v, 1, batch_size);
            v_t_final = ggml_reshape_2d(ctx, v_t_repeated, S_v, H_v * batch_size);
            
            // beta_t: [1, H_k, batch_size] -> [1, H_k, batch_size, 1] -> repeat -> [1, H_v, batch_size, 1]
            struct ggml_tensor * beta_t_4d = ggml_reshape_4d(ctx, beta_t_reshaped, 1, H_k, 1, batch_size);
            struct ggml_tensor * beta_t_repeated = ggml_repeat_4d(ctx, beta_t_4d, 1, H_v, 1, batch_size);
            beta_t_final = ggml_reshape_2d(ctx, beta_t_repeated, 1, H_v * batch_size);
        }
        
        // Ensure kv_mem has correct dimensions for subtraction
        // kv_mem dimensions from trace: [128, 16384, 1, 1]
        // We need to reshape it to match v_t_final: [128, 16, 1, 1]
        
        // First, let's reshape kv_mem to the correct dimensions
        struct ggml_tensor * kv_mem_reshaped;
        if (kv_mem->ne[0] == S_v && kv_mem->ne[1] == H_v * batch_size) {
            // Perfect match
            kv_mem_reshaped = kv_mem;
        } else if (kv_mem->ne[0] == S_v) {
            // We have the right first dimension, need to fix the second dimension
            kv_mem_reshaped = ggml_view_2d(ctx, kv_mem, S_v, H_v * batch_size, kv_mem->nb[1], 0);
        } else {
            // Handle other dimension mismatches
            report_tensor_size("kv_mem_before_reshape", kv_mem);
            kv_mem_reshaped = ggml_reshape_2d(ctx, kv_mem, S_v, H_v * batch_size);
        }
        kv_mem_reshaped = ggml_cont(ctx, kv_mem_reshaped);
        report_tensor_size("kv_mem_reshaped", kv_mem_reshaped);
        
        // Now ensure kv_mem_reshaped has the same dimensions as v_t_final
        struct ggml_tensor * kv_mem_final;
        if (kv_mem_reshaped->ne[0] == v_t_final->ne[0] && kv_mem_reshaped->ne[1] == v_t_final->ne[1]) {
            kv_mem_final = kv_mem_reshaped;
        } else {
            // Use repeat to match dimensions if they're compatible
            kv_mem_final = ggml_repeat(ctx, kv_mem_reshaped, v_t_final);
        }
        report_tensor_size("kv_mem_final", kv_mem_final);
        
        // Compute delta = (v_t - kv_mem) * beta_t
        struct ggml_tensor * delta = ggml_mul(ctx, ggml_sub(ctx, v_t_final, kv_mem_final), beta_t_final);
        report_tensor_size("delta", delta);
        
        // Update state: state = state + outer(k_t, delta)
        struct ggml_tensor * delta_reshaped = ggml_reshape_2d(ctx, delta, S_v, H_v * batch_size);
        report_tensor_size("delta_reshaped", delta_reshaped);
        
        // Handle the outer product for all heads and batches
        // We need to compute outer(k_t, delta) where:
        // k_t is [S_k * H_k, batch_size] -> reshape to [S_k, H_k * batch_size]
        // delta is [S_v, H_v * batch_size]
        // For outer product, we want k_t @ delta^T
        
        // First, handle head dimension mismatch for k_t (reuse existing k_t_final variable)
        if (H_k == H_v) {
            k_t_final = k_t_reshaped;
        } else {
            // Need to repeat k along the head dimension to match H_v
            int64_t repeat_factor = H_v / H_k;
            GGML_ASSERT(H_v % H_k == 0);
            
            // Reshape to separate repeat dimension: [S_k, 1, H_k, batch_size]
            k_t_final = ggml_reshape_3d(ctx, k_t_reshaped, S_k, 1, H_k * batch_size);
            report_tensor_size("k_t_final_reshape1", k_t_final);
            
            // Repeat along the new dimension: [S_k, repeat_factor, H_k, batch_size]
            k_t_final = ggml_repeat_4d(ctx, k_t_final, S_k, repeat_factor, H_k, batch_size);
            report_tensor_size("k_t_final_repeat", k_t_final);
            
            // Reshape back: [S_k, H_v * batch_size]
            k_t_final = ggml_reshape_2d(ctx, k_t_final, S_k, H_v * batch_size);
            report_tensor_size("k_t_final_reshape2", k_t_final);
        }
        
        // Make k_t_final contiguous
        k_t_final = ggml_cont(ctx, k_t_final);
        report_tensor_size("k_t_final_cont", k_t_final);
        
        // Handle dimension mismatch between S_k and S_v
        struct ggml_tensor * k_t_for_outer;
        if (S_k == S_v) {
            k_t_for_outer = k_t_final;
        } else if (S_k < S_v) {
            // Pad k_t to match S_v
            struct ggml_tensor * padding = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v - S_k, H_v * batch_size);
            report_tensor_size("k_t_padding", padding);
            k_t_for_outer = ggml_concat(ctx, k_t_final, padding, 0);
            report_tensor_size("k_t_for_outer_padded", k_t_for_outer);
        } else {
            // Truncate k_t to match S_v
            k_t_for_outer = ggml_view_2d(ctx, k_t_final, S_v, H_v * batch_size, k_t_final->nb[1], 0);
            report_tensor_size("k_t_for_outer_truncated", k_t_for_outer);
        }
        
        // Make sure k_t_for_outer is contiguous
        k_t_for_outer = ggml_cont(ctx, k_t_for_outer);
        report_tensor_size("k_t_for_outer_cont", k_t_for_outer);
        
        // Compute outer product: k_t_for_outer @ delta_reshaped^T
        // k_t_for_outer: [S_v, H_v * batch_size]
        // delta_reshaped: [S_v, H_v * batch_size]
        // For outer product, we want: k_t_for_outer @ delta_reshaped^T
        
        // We need to satisfy ggml_can_mul_mat requirements:
        // t0->ne[0] == t1->ne[0] (first dimensions must be equal)
        // t1->ne[2]%t0->ne[2] == 0 (broadcastable along 3rd dimension)
        // t1->ne[3]%t0->ne[3] == 0 (broadcastable along 4th dimension)
        
        // First, reshape k_t_for_outer to [S_v, H_v * batch_size, 1, 1]
        struct ggml_tensor * k_t_reshaped_4d = ggml_reshape_4d(ctx, k_t_for_outer, S_v, H_v, 1, batch_size);
        report_tensor_size("k_t_reshaped_4d", k_t_reshaped_4d);
        
        // Transpose delta_reshaped to get [H_v * batch_size, S_v]
        struct ggml_tensor * delta_transposed = ggml_transpose(ctx, delta_reshaped);
        report_tensor_size("delta_transposed", delta_transposed);
        
        // Make delta_transposed contiguous before reshaping
        delta_transposed = ggml_cont(ctx, delta_transposed);
        report_tensor_size("delta_transposed_cont", delta_transposed);
        
        // Reshape delta_transposed to [H_v * batch_size, S_v, 1, 1]
        struct ggml_tensor * delta_reshaped_4d = ggml_reshape_4d(ctx, delta_transposed, H_v, S_v, 1, batch_size);
        report_tensor_size("delta_reshaped_4d", delta_reshaped_4d);
        
        // For outer product k @ delta^T, we need: [S_v, H_v * batch_size] @ [H_v * batch_size, S_v] = [S_v, S_v]
        // But GGML requires the first dimensions to be equal for matrix multiplication
        // So we need to transpose the first tensor: k_t_reshaped_4d^T @ delta_reshaped_4d
        // [H_v * batch_size, S_v] @ [H_v * batch_size, S_v] - this won't work
        
        // Instead, we need to do: delta_reshaped_4d^T @ k_t_reshaped_4d^T
        // But GGML doesn't allow transposed first argument, so we need to swap the order
        // and transpose the result if needed
        
        // Let's do: delta_reshaped_4d^T @ k_t_reshaped_4d
        // [S_v, H_v * batch_size] @ [S_v, H_v * batch_size] - this won't work either
        
        // The correct approach is: k_t_reshaped_4d @ delta_reshaped_4d^T
        // But we need to make the first dimensions equal by transposing k_t_reshaped_4d
        struct ggml_tensor * k_t_transposed = ggml_transpose(ctx, k_t_reshaped_4d);
        report_tensor_size("k_t_transposed", k_t_transposed);
        
        // Now we can do: k_t_transposed @ delta_reshaped_4d
        // [H_v * batch_size, S_v] @ [H_v * batch_size, S_v] - still won't work
        
        // Let's try a different approach: use the transpose of the result
        // We want: k @ delta^T = (delta @ k^T)^T
        struct ggml_tensor * temp_product = ggml_mul_mat(ctx, delta_reshaped_4d, k_t_transposed);
        report_tensor_size("temp_product", temp_product);
        
        // Transpose the result to get the final outer product
        struct ggml_tensor * outer_product_raw = ggml_transpose(ctx, temp_product);
        report_tensor_size("outer_product_raw", outer_product_raw);
        
        // Make outer_product_raw contiguous before reshaping
        struct ggml_tensor * outer_product_cont = ggml_cont(ctx, outer_product_raw);
        report_tensor_size("outer_product_cont", outer_product_cont);
        
        // Reshape to 2D: [S_v, S_v]
        struct ggml_tensor * outer_product = ggml_reshape_2d(ctx, outer_product_cont, S_v, S_v);
        report_tensor_size("outer_product", outer_product);
        
        // Now we need to reshape outer_product to match state_flat dimensions
        // outer_product: [S_v, S_v] -> reshape to [S_v * S_v, H_v * batch_size]
        struct ggml_tensor * outer_product_reshaped;
        if (outer_product->ne[0] == S_v && outer_product->ne[1] == S_v) {
            // Perfect match for a single head/sequence
            outer_product_reshaped = ggml_reshape_2d(ctx, outer_product, S_v * S_v, 1);
        } else {
            // Handle whatever dimensions we got
            outer_product_reshaped = ggml_reshape_2d(ctx, outer_product,
                                                    outer_product->ne[0] * outer_product->ne[1], 1);
        }
        report_tensor_size("outer_product_reshaped", outer_product_reshaped);
        
        // Repeat outer_product_reshaped to match the number of heads and batches
        struct ggml_tensor * outer_product_repeated = ggml_repeat(ctx, outer_product_reshaped, state_flat);
        report_tensor_size("outer_product_repeated", outer_product_repeated);
        
        // Update state
        state_flat = ggml_add(ctx, state_flat, outer_product_repeated);
        report_tensor_size("state_flat_updated", state_flat);
        
        // Compute output = current_state @ q_t^T for all heads and batches
        // Simplified approach: follow Python implementation more closely
        // In Python: output = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        // This means: for each batch and head, multiply state by q_t and sum over the last dimension
        
        // First, let's work with the original q_t (already processed to match H_v)
        struct ggml_tensor * q_t_final = q_t;
        report_tensor_size("q_t_final", q_t_final);
        
        // Make q_t_final contiguous for matrix operations
        q_t_final = ggml_cont(ctx, q_t_final);
        report_tensor_size("q_t_final_cont", q_t_final);
        
        // For the output computation, we want: (state * q_t.unsqueeze(-1)).sum(dim=-2)
        // This is equivalent to: state @ q_t^T where q_t is reshaped appropriately
        
        // Simple approach: reshape q_t to [S_k, H_v * batch_size] and state to [S_v * S_v, H_v * batch_size]
        // Then compute: state^T @ q_t
        // But we need to handle the GGML requirements
        
        // Make state_flat contiguous
        struct ggml_tensor * state_flat_cont = ggml_cont(ctx, state_flat);
        report_tensor_size("state_flat_cont", state_flat_cont);
        
        // Reshape q_t to [S_k, H_v * batch_size] for matrix multiplication
        struct ggml_tensor * q_t_matrix = ggml_reshape_2d(ctx, q_t_final, S_k, H_v * batch_size);
        report_tensor_size("q_t_matrix", q_t_matrix);
        
        // Now we want to compute: state_flat_cont^T @ q_t_matrix
        // state_flat_cont: [S_v * S_v, H_v * batch_size] = [16384, 16]
        // q_t_matrix: [S_k, H_v * batch_size] = [128, 16]
        
        // For GGML, we need: q_t_matrix^T @ state_flat_cont^T
        // But GGML doesn't allow transposed first argument, so we use the property: A @ B = (B^T @ A^T)^T
        
        // Transpose q_t_matrix to get [H_v * batch_size, S_k] = [16, 128]
        struct ggml_tensor * q_t_matrix_transposed = ggml_transpose(ctx, q_t_matrix);
        report_tensor_size("q_t_matrix_transposed", q_t_matrix_transposed);
        
        // Transpose state_flat_cont to get [H_v * batch_size, S_v * S_v] = [16, 16384]
        struct ggml_tensor * state_flat_transposed = ggml_transpose(ctx, state_flat_cont);
        report_tensor_size("state_flat_transposed", state_flat_transposed);
        
        // Now we can do: q_t_matrix_transposed @ state_flat_transposed
        // [16, 128] @ [16, 16384] - this won't work because first dimensions don't match
        
        // Instead, let's do: state_flat_transposed^T @ q_t_matrix_transposed^T
        // But we need to transpose both again
        struct ggml_tensor * q_t_matrix_final = ggml_transpose(ctx, q_t_matrix_transposed);
        report_tensor_size("q_t_matrix_final", q_t_matrix_final);
        
        struct ggml_tensor * state_flat_final = ggml_transpose(ctx, state_flat_transposed);
        report_tensor_size("state_flat_final", state_flat_final);
        
        // Now we can do: q_t_matrix_final @ state_flat_final
        // [128, 16] @ [16384, 16] - this won't work either
        
        // Let me try a different approach: use element-wise multiplication and sum
        // We want: (state * q_t.unsqueeze(-1)).sum(dim=-2)
        
        // First, reshape q_t to broadcast with state
        struct ggml_tensor * q_t_broadcast = ggml_repeat(ctx, q_t_final, state_flat_cont);
        report_tensor_size("q_t_broadcast", q_t_broadcast);
        
        // Element-wise multiplication
        struct ggml_tensor * state_q_product = ggml_mul(ctx, state_flat_cont, q_t_broadcast);
        report_tensor_size("state_q_product", state_q_product);
               
        // Let's reshape to separate the dimensions we want to sum over
        
        // Reshape state_q_product to [S_v * S_v, H_v, batch_size]
        struct ggml_tensor * state_q_3d = ggml_reshape_3d(ctx, state_q_product, S_v * S_v, H_v, batch_size);
        report_tensor_size("state_q_3d", state_q_3d);
        // Ensure contiguous layout so byte-strides are consistent for subsequent views/slices.
        state_q_3d = ggml_cont(ctx, state_q_3d);
        report_tensor_size("state_q_3d_cont", state_q_3d);
        
        // Sum over the H_v dimension (axis 1)
        // Create a proper ones vector: ggml_new_tensor_1d already creates a zero-filled tensor,
        // so ggml_exp on it will produce ones (exp(0) = 1).
        struct ggml_tensor * ones_vector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
        ones_vector = ggml_exp(ctx, ones_vector);      // exp(0) = 1
        report_tensor_size("ones_vector", ones_vector);
        
        // Reshape to [H_v, 1] for matrix multiplication
        struct ggml_tensor * ones_col = ggml_reshape_2d(ctx, ones_vector, H_v, 1);
        report_tensor_size("ones_col", ones_col);
        
        // Prepare per-batch results
        struct ggml_tensor * output_parts[batch_size];
        for (int64_t b = 0; b < batch_size; b++) {
            // Extract slice for this batch: [S_v * S_v, H_v]
            // Use the contiguous state_q_3d so nb and offsets are reliable.
            struct ggml_tensor * batch_slice = ggml_view_3d(ctx, state_q_3d, S_v * S_v, H_v, 1,
                                                           state_q_3d->nb[1], state_q_3d->nb[2], b * state_q_3d->nb[2]);
            batch_slice = ggml_cont(ctx, batch_slice);
            report_tensor_size("batch_slice", batch_slice);
            
            // Multiply by ones and sum across H_v:
            // ones_col: [H_v, 1], batch_slice^T: [H_v, S_v * S_v] -> ones_col @ batch_slice^T = [1, S_v * S_v]
            struct ggml_tensor * batch_slice_t = ggml_transpose(ctx, batch_slice);
            report_tensor_size("batch_slice_t", batch_slice_t);
            struct ggml_tensor * batch_sum = ggml_mul_mat(ctx, ones_col, batch_slice_t);
            report_tensor_size("batch_sum", batch_sum);
            
            // Reshape [1, S_v*S_v] -> [S_v, S_v]
            struct ggml_tensor * batch_result = ggml_reshape_2d(ctx, batch_sum, S_v, S_v);
            report_tensor_size("batch_result", batch_result);
            output_parts[b] = batch_result;
        }
        
        // Concatenate results from all batches into [S_v * S_v, batch_size]
        struct ggml_tensor * output_concat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v * S_v, batch_size);
        for (int64_t b = 0; b < batch_size; b++) {
            struct ggml_tensor * batch_output = ggml_view_2d(ctx, output_concat, S_v * S_v, 1,
                                                            output_concat->nb[1], b * output_concat->nb[1]);
            batch_output = ggml_cpy(ctx, output_parts[b], batch_output);
        }
        
        // Reshape concatenated result to [S_v, S_v] for this token (batch_size typically 1)
        struct ggml_tensor * output_t_reshaped = ggml_reshape_2d(ctx, output_concat, S_v, S_v);
        struct ggml_tensor * output_t = ggml_cont(ctx, output_t_reshaped);
        report_tensor_size("output_t", output_t);
              
        // Store output for this token
        struct ggml_tensor * output_slice = ggml_view_3d(ctx, output, S_v, S_v, batch_size,
                                                        output->nb[1], output->nb[2], t * output->nb[2]);
        report_tensor_size("output_slice", output_slice);
        output_slice = ggml_cpy(ctx, output_t, output_slice);
        report_tensor_size("output_slice_copied", output_slice);
    }
    
    struct ggml_tensor * result = ggml_concat(ctx, output, new_state, 2);
    report_tensor_size("result_final", result);
    return result;
}
// ggml_rwkv_wkv7
