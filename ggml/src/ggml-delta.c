#include "ggml.h"
#include "ggml-delta.h"
#include "ggml-impl.h"

static void report_tensor_size(const char * tensor_name, const struct ggml_tensor * tensor) {
#ifdef HAVE_DEBUG_DELTA_NET
    GGML_LOG_INFO("[%s] tensor size is [%lu, %lu, %lu, %lu], strides [%lu, %lu, %lu, %lu]\n", 
        tensor_name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
#endif
}

// ggml_delta_net
struct ggml_tensor * ggml_delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * q,
        struct ggml_tensor  * g,
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
    const int64_t n_tokens = k->ne[2];  
    const int64_t n_seqs = k->ne[3];
    
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(q->ne[2] == n_tokens);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[1] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == n_seqs && state->ne[3] == n_tokens);
    
    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens);
       
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
       
    // Beta sigmoid
    struct ggml_tensor * beta_sigmoid = ggml_sigmoid(ctx, beta);
    report_tensor_size("beta_sigmoid", beta_sigmoid);

    // Gate calculations are done elsewhere in llama-model.cpp

    struct ggml_tensor * q_broadcast = q;
    struct ggml_tensor * k_broadcast = k;
    
    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (H_k != H_v) {
        GGML_ASSERT(H_v % H_k == 0);
        int64_t repeat_factor = H_v / H_k;
        
        q_broadcast = ggml_cont_4d(ctx, q, S_k, n_tokens, H_k, n_seqs);
        report_tensor_size("q_broadcast_reshape1", q_broadcast);
        k_broadcast = ggml_cont_4d(ctx, k, S_k, n_tokens, H_k, n_seqs);
        report_tensor_size("k_broadcast_reshape1", k_broadcast);
        
        q_broadcast = ggml_repeat_4d(ctx, q_broadcast, S_k, n_tokens * repeat_factor, H_k, n_seqs);
        report_tensor_size("q_broadcast_repeat", q_broadcast);
        k_broadcast = ggml_repeat_4d(ctx, k_broadcast, S_k, n_tokens * repeat_factor, H_k, n_seqs);
        report_tensor_size("k_broadcast_repeat", k_broadcast);
        
        q_broadcast = ggml_reshape_4d(ctx, q_broadcast, S_k, H_v, n_seqs, n_tokens);
        report_tensor_size("q_broadcast_reshape2", q_broadcast);
        k_broadcast = ggml_reshape_4d(ctx, k_broadcast, S_k, H_v, n_seqs, n_tokens);
        report_tensor_size("k_broadcast_reshape2", k_broadcast);
    }

    struct ggml_tensor * v_reshape = ggml_cont_4d(ctx, v, S_v, H_v, n_seqs, n_tokens);
    report_tensor_size("v_reshape", v_reshape);
    struct ggml_tensor * g_reshape = ggml_cont_4d(ctx, g, S_v, H_v, n_seqs, n_tokens);
    report_tensor_size("g_reshape", g_reshape);
    struct ggml_tensor * beta_broadcast = ggml_cont_4d(ctx, beta, 1, H_v, n_seqs, n_tokens);
    report_tensor_size("beta_broadcast", beta_broadcast);
    struct ggml_tensor * state_broadcast = ggml_cont(ctx, state);
    report_tensor_size("state_broadcast", state_broadcast);
    
    return ggml_delta_net_op(ctx, q_broadcast, k_broadcast, v_reshape, g_reshape, beta_broadcast, state_broadcast, use_qk_l2norm, scale);
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
    
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));
    
    const int64_t S_k = q->ne[0];  
    const int64_t H_k = q->ne[1];  
    const int64_t n_seq = q->ne[2];  
    const int64_t n_tokens = q->ne[3];
    
    const int64_t S_v = v->ne[0];  
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(H_k == H_v); // we broadcasted the tensors in the main function to guarantee this
    
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_v && k->ne[2] == n_seq && k->ne[3] == n_tokens);
    GGML_ASSERT(v->ne[1] == H_v && v->ne[2] == n_seq && v->ne[3] == n_tokens);
    GGML_ASSERT(g->ne[0] == S_v && g->ne[1] == H_v && g->ne[2] == n_seq && g->ne[3] == n_tokens);
    GGML_ASSERT(beta->ne[0] == 1 && beta->ne[1] == H_v && beta->ne[2] == n_seq && beta->ne[3] == n_tokens);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v * H_v && state->ne[2] == n_seq && state->ne[3] == n_tokens);
       
    struct ggml_tensor * new_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v * H_v, n_seq, n_tokens);
    
    new_state = ggml_cpy(ctx, state, new_state);
    report_tensor_size("new_state_copied", new_state);
    
    if (use_qk_l2norm) {
        q = ggml_l2_norm(ctx, q, 1e-6f);
        report_tensor_size("q_l2norm", q);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        report_tensor_size("k_l2norm", k);
    }
    
    q = ggml_scale(ctx, q, scale);
    report_tensor_size("q_scaled", q);
    
    struct ggml_tensor * state_flat = ggml_reshape_2d(ctx, new_state, S_v * S_v * H_v, n_seq * n_tokens);
    report_tensor_size("state_flat", state_flat);
                                  
    struct ggml_tensor * state_decay = ggml_mul(ctx, state, g);
    report_tensor_size("state_decay", state_decay);
               
    struct ggml_tensor * kv_mem_presum = ggml_mul(ctx, state_decay, k);
    report_tensor_size("kv_mem_presum", kv_mem_presum);

    // Gotta do some squeezing here...
    struct ggml_tensor * kv_mem_presum_squeeze = ggml_reshape_4d(ctx, kv_mem_presum, S_v, S_v, H_v, n_seq * n_tokens);
    report_tensor_size("kv_mem_presum_sequeeze", kv_mem_presum_squeeze);

    struct ggml_tensor * kv_mem = ggml_permute(ctx, ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, kv_mem_presum_squeeze, 1, 2, 0, 3))), 2, 0, 1, 3);
    report_tensor_size("kv_mem", kv_mem);

    struct ggml_tensor * kv_mem_reshape = ggml_reshape_4d(ctx, kv_mem, S_v, S_v, n_seq, n_tokens);
    report_tensor_size("kv_mem_reshape", kv_mem_reshape);
                
    struct ggml_tensor * delta = ggml_mul(ctx, ggml_sub(ctx, kv_mem_reshape, v), beta);
    report_tensor_size("delta", delta);

    struct ggml_tensor * delta_kt = ggml_mul(ctx, delta, k);
    report_tensor_size("delta_kt", delta_kt);

    struct ggml_tensor * state_plus_k_delta = ggml_add(ctx, state_decay, delta_kt);
    report_tensor_size("state_plus_k_delta", state_plus_k_delta);

    struct ggml_tensor * state_q = ggml_mul(ctx, state_plus_k_delta, q);
    report_tensor_size("state_q", state_q);

    // And here...
    state_q = ggml_reshape_4d(ctx, state_q, S_v, S_v, H_v, n_seq * n_tokens);
    struct ggml_tensor * output = ggml_permute(ctx, ggml_sum_rows(ctx, state_q), 2, 0, 1, 3);
    output = ggml_reshape_4d(ctx, output, S_v, H_v, n_seq, n_tokens);
    report_tensor_size("output", output);
    
    struct ggml_tensor * result = ggml_concat(ctx, output, state_plus_k_delta, 1);
    report_tensor_size("result_final", result);
    return result;
}


