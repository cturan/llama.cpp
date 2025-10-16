#pragma once

#include "../llama-graph.h"
#include "../llama-model.h"
#include "llm_graph_context_mamba.h"

#include <cmath>

struct llm_build_qwen3next : public llm_graph_context_mamba {
    llm_build_qwen3next(const llama_model & model, const llm_graph_params & params);

private:
    // delta_net
    struct ggml_tensor * delta_net(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 eps_norm,
        const int             il);

    // delta_net_recurrent
    struct ggml_tensor * delta_net_recurrent(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * g,
        struct ggml_tensor  * beta,
        struct ggml_tensor  * state,
        bool                  use_qk_l2norm,
        float                 eps_norm,
        const int             il);

    ggml_tensor * build_qwen3next_attention_layer(ggml_tensor *             cur,
                                                  ggml_tensor *             inp_pos,
                                                  llm_graph_input_attn_kv * inp_attn,
                                                  const llama_model &       model,
                                                  const int64_t             n_embd_head,
                                                  const int                 il);

    ggml_tensor * build_qwen3next_linear_attn_layer(llm_graph_input_rs * inp,
                                                    ggml_tensor *        cur,
                                                    const llama_model &  model,
                                                    const llama_ubatch & ubatch,
                                                    int                  il);

    ggml_tensor * build_layer_ffn(ggml_tensor * cur, const llama_model & model, const int il);

    ggml_tensor * build_q3n_norm(struct ggml_tensor * input, struct ggml_tensor * weights, int layer);
    ggml_tensor * build_q3n_gated_norm(struct ggml_tensor * input, struct ggml_tensor * weights, struct ggml_tensor * gate, int layer);

};