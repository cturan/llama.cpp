#include "llm_build_qwen3next.h"

#include <cmath>

// Implementation of depthwise 1D convolution using F32 to avoid F16 limitations
static ggml_tensor* ggml_conv_1d_dw_f32(
        ggml_context * ctx,
        ggml_tensor  * kernel,
        ggml_tensor  * input,
        int           stride,
        int           padding,
        int           dilation) {
    // Following the pattern from ggml_conv_1d_dw but using F32
    // Reshape input from [length, channels, batch, dummy] to [length, 1, channels, batch]
    ggml_tensor* reshaped_input = ggml_reshape_4d(ctx, input, input->ne[0], 1, input->ne[1], input->ne[2]);

    // Apply im2col with F32 destination type to avoid F16 requirement
    ggml_tensor* im2col_result = ggml_im2col(ctx, kernel, reshaped_input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);

    // Now multiply: im2col_result * kernel (following the exact pattern from ggml_conv_1d_dw)
    // In ggml_conv_1d_dw: ggml_mul_mat(ctx, im2col, a) where a is the kernel
    ggml_tensor* mul_result = ggml_mul_mat(ctx, im2col_result, kernel);

    // Reshape the result following ggml_conv_1d_dw: [result->ne[0], result->ne[2], 1]
    ggml_tensor* output_3d = ggml_reshape_3d(ctx, mul_result, mul_result->ne[0], mul_result->ne[2], 1);
    return output_3d;
}

llm_build_qwen3next::llm_build_qwen3next(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;
        cur = build_q3n_norm(inpL, model.layers[il].attn_norm, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_qwen3next_linear_attn_layer(inp->get_recr(), cur, model, ubatch, il);
        } else {
            // Full attention layer
            cur = build_qwen3next_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il);
        }
        // Post-attention norm
        cur = build_q3n_norm(cur, model.layers[il].attn_post_norm, il);
        cb(cur, "attn_post_norm", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // FFN layer (MoE or dense)
        cur = build_layer_ffn(cur, model, il);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_q3n_norm(cur, model.output_norm, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    ggml_set_output(cur);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

struct ggml_tensor * llm_build_qwen3next::build_q3n_norm(struct ggml_tensor * input, struct ggml_tensor * weights, int layer) {
    ggml_tensor * input_norm = ggml_scale_bias(ctx0, weights, 1.0f, 1.0f);
    return build_norm(input, input_norm, nullptr, LLM_NORM_RMS, layer);
}

ggml_tensor * llm_build_qwen3next::build_qwen3next_attention_layer(ggml_tensor *             cur,
                                                                   ggml_tensor *             inp_pos,
                                                                   llm_graph_input_attn_kv * inp_attn,
                                                                   const llama_model &       model,
                                                                   const int64_t             n_embd_head,
                                                                   const int                 il) {
    ggml_tensor * gate = build_lora_mm(model.layers[il].wq_gate, cur);

    // compute Q and K and RoPE them
    struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur, "Qcur", il);

    struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply Q/K normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
    cb(Kcur, "Qcur_normed", il);
    cb(Kcur, "Kcur_normed", il);

    // Apply RoPE
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                         attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor,
                         attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp_attn, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

    // Apply gating
    cur = ggml_cont(ctx0, ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate)));
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen3next::build_qwen3next_linear_attn_layer(llm_graph_input_rs * inp,
                                                                     ggml_tensor *        cur,
                                                                     const llama_model &  model,
                                                                     const llama_ubatch & ubatch,
                                                                     int                  il) {
    // Gated Delta Net implementation using the new ggml_delta_net function
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t n_heads  = hparams.ssm_dt_rank;
    const int64_t head_dim = d_inner / n_heads;
    const int64_t n_seqs   = ubatch.n_seqs;

    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t head_v_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_tokens     = ubatch.n_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, cur);
    cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

    ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
    cb(mixed_ba, "linear_attn_mixed_ba", il);

    int64_t       qkvz_new_dim        = 2 * head_k_dim + 2 * head_v_dim * num_v_heads / num_k_heads;
    ggml_tensor * mixed_qkvz_reshaped = ggml_cont_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_tokens, n_seqs);

    // Reshape mixed_ba: [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    int64_t       ba_new_dim        = 2 * num_v_heads / num_k_heads;
    ggml_tensor * mixed_ba_reshaped = ggml_cont_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_tokens, n_seqs);

    // Split mixed_ba into b and a (beta and alpha parameters)
    int64_t split_sizes_ba[2] = {
        num_v_heads / num_k_heads,  // beta size
        num_v_heads / num_k_heads   // alpha size
    };

    ggml_tensor * b = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
    cb(b, "b", il);

    ggml_tensor * a = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_tokens, n_seqs,
                        mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 
                        split_sizes_ba[0] * ggml_element_size(mixed_ba_reshaped));
    cb(a, "a", il);

    // Reshape b and a to merge head dimensions: [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    ggml_tensor * beta  = ggml_cont_3d(ctx0, b, num_v_heads, n_tokens, n_seqs);
    ggml_tensor * alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_tokens, n_seqs);

    GGML_ASSERT(ggml_nelements(beta) + ggml_nelements(alpha) == ggml_nelements(mixed_ba));

    ggml_tensor * alpha_softplus = softplus(alpha, model.layers[il].ssm_dt);
    cb(alpha_softplus, "a_softplus", il);
    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

        // Split mixed_qkvz into query, key, value, z
    int64_t split_sizes_qkvz[4] = {
        head_k_dim,                              // query size
        head_k_dim,                              // key size
        head_v_dim * num_v_heads / num_k_heads,  // value size
        head_v_dim * num_v_heads / num_k_heads   // z size
    };

    ggml_tensor * query = ggml_cont(ctx0, ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_tokens, n_seqs, 
                                            mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0));
    cb(query, "q", il);

    ggml_tensor * key = ggml_cont(ctx0, ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        split_sizes_qkvz[0] * sizeof(float)));
    cb(key, "k", il);

    ggml_tensor * value = ggml_cont(ctx0, ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * sizeof(float)));
    cb(value, "v", il);

    ggml_tensor * z = ggml_cont(ctx0, ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * sizeof(float)));
    cb(z, "z", il);

    // Reshape value and z to merge head dimensions: [batch, seq_len, num_k_heads, head_v_dim*num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads, head_v_dim]
    ggml_tensor * value_reshaped =
        ggml_reshape_4d(ctx0, ggml_cont(ctx0, value), head_v_dim, num_v_heads, n_tokens, n_seqs);
    ggml_tensor * z_reshaped = ggml_reshape_4d(ctx0, ggml_cont(ctx0, z), head_v_dim, num_v_heads, n_tokens, n_seqs);

    GGML_ASSERT(ggml_nelements(query) + ggml_nelements(key) + ggml_nelements(value_reshaped) +
                    ggml_nelements(z_reshaped) ==
                ggml_nelements(mixed_qkvz));

    // After creating query, key, and value_reshaped, reshape each to flatten the head dimensions
    // query: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * query_flat = ggml_reshape_3d(ctx0, query, head_k_dim * num_k_heads, n_tokens, n_seqs);
    cb(query_flat, "query_flat", il);

    // key: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
    ggml_tensor * key_flat = ggml_reshape_3d(ctx0, key, head_k_dim * num_k_heads, n_tokens, n_seqs);
    cb(key_flat, "key_flat", il);

    // value_reshaped: [head_v_dim, num_v_heads, n_tokens, n_seqs] -> [head_v_dim * num_v_heads, n_tokens, n_seqs]
    ggml_tensor * value_flat = ggml_reshape_3d(ctx0, value_reshaped, head_v_dim * num_v_heads, n_tokens, n_seqs);
    cb(value_flat, "value_flat", il);

    // Now concatenate along the feature dimension (dim 0) to get [conv_dim, n_tokens, n_seqs]
    ggml_tensor * qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
    qkv_mixed = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
    qkv_mixed = ggml_permute(ctx0, qkv_mixed, 1, 0, 2, 3);
    cb(qkv_mixed, "qkv_mixed_concatenated", il);

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    conv_kernel = ggml_permute(ctx0, conv_kernel, 0, 2, 1, 3);
    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);    
    cb(conv_input, "conv_input", il);

    // Apply convolution
    ggml_tensor * conv_output = ggml_conv_1d_dw_f32(ctx0, conv_kernel, conv_input, 1, conv_kernel_size - 1, n_seqs);
    cb(conv_output, "conv_output_raw", il);

    // Remove the padding
    ggml_tensor * conv_output_no_padding = ggml_view_4d(ctx0, conv_output, conv_output->ne[0] - (conv_kernel_size - 1), conv_output->ne[1], conv_output->ne[2], conv_output->ne[3],
                                                        conv_output->nb[1], conv_output->nb[2], conv_output->nb[3], 
                                                        (conv_kernel_size - 1) * ggml_element_size(conv_output));
    cb(conv_output_no_padding, "conv_output_no_padding", il);

    // Take only the first (n_tokens * n_seqs) values
    ggml_tensor * conv_output_proper = ggml_view_4d(ctx0, conv_output_no_padding, n_tokens * n_seqs, conv_output_no_padding->ne[1], conv_output_no_padding->ne[2], conv_output_no_padding->ne[3],
                                                        conv_output_no_padding->nb[1], conv_output_no_padding->nb[2], conv_output_no_padding->nb[3], 0);
    cb(conv_output_proper, "conv_output_proper", il);

    conv_output_proper = ggml_permute(ctx0, conv_output_proper, 0, 1, 3, 2);
    conv_output_proper = ggml_cont_4d(ctx0, conv_output_proper, qkv_dim, 1, n_tokens, n_seqs);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, qkv_dim, n_seqs, conv_input->nb[1], conv_input->nb[2],
                     n_seq_tokens * conv_input->nb[0]);

    ggml_build_forward_expand(gf,
                              ggml_cpy(ctx0, last_conv_states,
                                       ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * qkv_dim * n_seqs,
                                                    mctx_cur->get_head() * (conv_kernel_size - 1) * qkv_dim *
                                                        ggml_element_size(conv_states_all))));
    cb(conv_states_all, "conv_states_updated", il);

    conv_output_proper = ggml_reshape_2d(ctx0, conv_output_silu, n_tokens * n_seqs, qkv_dim);
    cb(conv_output_proper, "conv_output_final", il);

    ggml_tensor * conv_transposed = ggml_transpose(ctx0, conv_output_proper);
    cb(conv_transposed, "conv_transposed", il);

    ggml_tensor * conv_qkv_mix = ggml_cont_2d(ctx0, conv_transposed, qkv_dim, n_tokens * n_seqs);
    cb(conv_qkv_mix, "conv_qkv_mix", il);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_tokens * n_seqs,
        conv_qkv_mix->nb[1], 0);
    cb(q_conv, "q_conv", il);
    ggml_tensor * k_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_k_dim * num_k_heads, n_tokens * n_seqs, 
        conv_qkv_mix->nb[1], head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(k_conv, "k_conv", il);
    ggml_tensor * v_conv = ggml_view_2d(ctx0, conv_qkv_mix, head_v_dim * num_v_heads, n_tokens * n_seqs, 
        conv_qkv_mix->nb[1], 2 * head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
    cb(v_conv, "v_conv", il);
    
    // Unsqueeze them
    q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_tokens, n_seqs);
    k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_tokens, n_seqs);
    v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_tokens, n_seqs);

    beta  = ggml_cont_4d(ctx0, b, 1, num_v_heads, n_tokens, n_seqs);
    alpha = ggml_cont_4d(ctx0, a, 1, num_v_heads, n_tokens, n_seqs);

    ggml_tensor * state = ggml_reshape_4d(ctx0, ssm_states_all, head_dim, head_dim * n_heads, 1, 1);
    gate = ggml_reshape_4d(ctx0, gate, 1, n_heads, n_tokens, n_seqs);

        // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        int64_t repeat_factor = num_v_heads / num_k_heads;

        q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, n_tokens, num_k_heads, n_seqs);
        k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, n_tokens, num_k_heads, n_seqs);

        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, n_tokens * repeat_factor, num_k_heads, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, n_tokens * repeat_factor, num_k_heads, n_seqs);

        // Fix dimension order: last two should be [tokens, batches]
        q_conv = ggml_reshape_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
        k_conv = ggml_reshape_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_tokens, n_seqs);
    }

    // Call the new ggml_delta_net function with the corrected flow
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(num_k_heads)) : hparams.f_attention_scale;
    ggml_tensor * attn_out = ggml_delta_net(ctx0, q_conv, k_conv, v_conv, gate, beta, state, true, kq_scale, hparams.f_norm_rms_eps);
    cb(attn_out, "attn_out", il);

    // The tensors were concatenated 1d, so we need to extract them 1d as well
    const int64_t output_flat_size = head_dim * n_heads * n_tokens * n_seqs;
    ggml_tensor * attn_out_1d =
        ggml_view_1d(ctx0, attn_out, output_flat_size, 0);
    cb(attn_out_1d, "attn_out_1d", il);
    
    ggml_tensor * attn_out_final = ggml_cont_4d(ctx0, attn_out_1d, head_dim, n_heads, n_tokens, n_seqs);
    cb(attn_out_final, "attn_out_final", il);
   
    // Extract the state part (second part of the concatenated tensor)
    // State starts after n_tokens elements along dimension 1
    const int64_t state_flat_size = head_dim * head_dim * n_heads * n_seqs;
    
    ggml_tensor * state_1d = ggml_view_1d(ctx0, attn_out, state_flat_size, output_flat_size * ggml_element_size(attn_out));
    cb(state_1d, "state_1d", il);
    
    ggml_tensor * new_state = ggml_reshape_4d(ctx0, state_1d, head_dim, head_dim, n_heads, n_seqs);
    cb(new_state, "new_state", il);

    // Update the recurrent states - we use the new_state directly since it's already the last state
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ssm_states_all));

    // Reshape both attn_out_final and z to 2D tensors for normalization
    // attn_out_final: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * attn_out_2d_final = ggml_reshape_2d(ctx0, ggml_cont(ctx0, attn_out_final), head_dim, n_heads * n_tokens * n_seqs);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_2d(ctx0, z_reshaped, head_dim, n_heads * n_tokens * n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    // This is Qwen3NextRMSNormGated which applies: RMSNorm(x) * silu(gate)
    ggml_tensor * attn_out_norm = build_norm(attn_out_2d_final, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
    cb(attn_out_norm, "attn_out_norm", il);

    // Apply silu gate: attn_out_norm * silu(z_2d)
    ggml_tensor * z_silu = ggml_silu(ctx0, z_2d);
    cb(z_silu, "z_silu", il);
    ggml_tensor * gated_output = ggml_mul(ctx0, attn_out_norm, z_silu);
    cb(gated_output, "gated_output", il);

    // Reshape back to original dimensions: [n_heads * n_tokens * n_seqs, head_dim] -> [head_dim, n_heads, n_tokens, n_seqs]
    ggml_tensor * gated_output_4d = ggml_reshape_4d(ctx0, gated_output, head_dim, n_heads, n_tokens, n_seqs);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, gated_output_4d, n_heads * head_dim, n_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_cont(ctx0, ggml_reshape_2d(ctx0, cur, n_embd, n_tokens));
    return cur;
}

ggml_tensor * llm_build_qwen3next::build_layer_ffn(ggml_tensor * cur, const llama_model & model, const int il) {
    // Check if this is an MoE layer
    if (model.layers[il].ffn_gate_inp != nullptr) {
        // MoE branch
        ggml_tensor * moe_out =
            build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                          model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, nullptr, n_expert,
                          n_expert_used, LLM_FFN_SILU, true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        cb(moe_out, "ffn_moe_out", il);

        // Add shared experts if present
        if (model.layers[il].ffn_up_shexp != nullptr) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur, model.layers[il].ffn_up_shexp, NULL, NULL, model.layers[il].ffn_gate_shexp, NULL, NULL,
                          model.layers[il].ffn_down_shexp, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    } else {
        // Dense FFN branch
        cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    }
    // Residual connection
    cur = ggml_add(ctx0, cur, cur);  // This should be the residual from before FFN
    cb(cur, "ffn_residual", il);

    return cur;
};

ggml_tensor * llm_build_qwen3next::softplus(ggml_tensor * alpha, ggml_tensor * dt_bias) {
    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, dt_bias);                // a + dt_bias
    ggml_tensor * alpha_exp      = ggml_exp(ctx0, alpha_biased);                  // exp(a + dt_bias)
    ggml_tensor * one_plus_exp   = ggml_scale_bias(ctx0, alpha_exp, 1.0f, 1.0f);  // 1 + exp(a + dt_bias)
    ggml_tensor * alpha_softplus = ggml_log(ctx0, one_plus_exp);                  // log(1 + exp(...))
    return alpha_softplus;
}
