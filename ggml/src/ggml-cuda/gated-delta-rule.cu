#include "common.cuh"
#include "gated-delta-rule.cuh"

static __device__ __forceinline__ float sigmoid_f32(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

template <typename T>
static __device__ __forceinline__ float load_f32(const T * __restrict__ p) {
    return (float) *p;
}

template <>
__device__ __forceinline__ float load_f32<half>(const half * __restrict__ p) {
    return __half2float(*p);
}

template <int K, int BV, typename T, typename S>
static __global__ void gated_delta_rule_fwd(
        const T     * __restrict__ q,       // [K, H, T, B]
        const T     * __restrict__ k,       // [K, H, T, B]
        const T     * __restrict__ v,       // [K, H, T, B]
        const T     * __restrict__ g,       // [H, T, B]
        const T     * __restrict__ beta,    // [H, T, B] (pre-sigmoid)
        const S     * __restrict__ s,       // [K, K, H, B] (row-major: s[row][col])
        float       * __restrict__ o,       // [K, H, T, B]
        float       * __restrict__ st,      // [K, K, H, B]
        const int                 H,
        const int                 T_len,
        const float               q_scale,
        const float               eps,
        const int64_t             q_nb1,
        const int64_t             q_nb2,
        const int64_t             q_nb3,
        const int64_t             k_nb1,
        const int64_t             k_nb2,
        const int64_t             k_nb3,
        const int64_t             v_nb1,
        const int64_t             v_nb2,
        const int64_t             v_nb3,
        const int64_t             g_nb1,
        const int64_t             g_nb2,
        const int64_t             beta_nb1,
        const int64_t             beta_nb2) {
    static_assert(K % WARP_SIZE == 0, "K must be divisible by warp size");
    static_assert(BV <= WARP_SIZE, "BV must be <= warp size");

    const int lane = threadIdx.x;
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh - b * H;
    const int v0 = v_tile * BV;

    constexpr int rows_per_thread = K / WARP_SIZE;
    float state[rows_per_thread][BV];

    const int64_t s_base = (int64_t) bh * K * K;

    // Load initial state
    #pragma unroll
    for (int rr = 0; rr < rows_per_thread; ++rr) {
        const int row = lane + rr * WARP_SIZE;
        #pragma unroll
        for (int cc = 0; cc < BV; ++cc) {
            const int col = v0 + cc;
            state[rr][cc] = col < K ? load_f32(s + s_base + (int64_t) row * K + col) : 0.0f;
        }
    }

    for (int t = 0; t < T_len; ++t) {
        const int64_t q_base    = (int64_t) h * q_nb1 + (int64_t) t * q_nb2 + (int64_t) b * q_nb3;
        const int64_t k_base    = (int64_t) h * k_nb1 + (int64_t) t * k_nb2 + (int64_t) b * k_nb3;
        const int64_t v_base    = (int64_t) h * v_nb1 + (int64_t) t * v_nb2 + (int64_t) b * v_nb3;
        const int64_t g_base    = (int64_t) h + (int64_t) t * g_nb1 + (int64_t) b * g_nb2;
        const int64_t beta_base = (int64_t) h + (int64_t) t * beta_nb1 + (int64_t) b * beta_nb2;
        const int64_t out_base  = (int64_t) K * (h + H * (t + T_len * b));

        float q_raw[rows_per_thread];
        float k_raw[rows_per_thread];
        float q_ss = 0.0f;
        float k_ss = 0.0f;

        #pragma unroll
        for (int rr = 0; rr < rows_per_thread; ++rr) {
            const int idx = lane + rr * WARP_SIZE;
            const float qv = load_f32(q + q_base + idx);
            const float kv = load_f32(k + k_base + idx);
            q_raw[rr] = qv;
            k_raw[rr] = kv;
            q_ss += qv * qv;
            k_ss += kv * kv;
        }

        q_ss = warp_reduce_sum(q_ss);
        k_ss = warp_reduce_sum(k_ss);

        const float q_inv = rsqrtf(q_ss + eps);
        const float k_inv = rsqrtf(k_ss + eps);

        float qn[rows_per_thread];
        float kn[rows_per_thread];

        #pragma unroll
        for (int rr = 0; rr < rows_per_thread; ++rr) {
            qn[rr] = q_raw[rr] * q_inv * q_scale;
            kn[rr] = k_raw[rr] * k_inv;
        }

        float gexp = 0.0f;
        float bsig = 0.0f;
        if (lane == 0) {
            gexp = expf(load_f32(g + g_base));
            bsig = sigmoid_f32(load_f32(beta + beta_base));
        }
        gexp = __shfl_sync(0xffffffff, gexp, 0);
        bsig = __shfl_sync(0xffffffff, bsig, 0);

        #pragma unroll
        for (int cc = 0; cc < BV; ++cc) {
            const int col = v0 + cc;
            if (col >= K) continue;

            float partial = 0.0f;
            #pragma unroll
            for (int rr = 0; rr < rows_per_thread; ++rr) {
                state[rr][cc] *= gexp;
                partial += state[rr][cc] * kn[rr];
            }
            const float dot_k = warp_reduce_sum(partial);

            float v_in = 0.0f;
            if (lane == cc) {
                v_in = load_f32(v + v_base + col);
            }
            v_in = __shfl_sync(0xffffffff, v_in, cc);

            const float v_new = bsig * (v_in - dot_k);

            float partial_o = 0.0f;
            #pragma unroll
            for (int rr = 0; rr < rows_per_thread; ++rr) {
                state[rr][cc] += kn[rr] * v_new;
                partial_o += state[rr][cc] * qn[rr];
            }
            const float out = warp_reduce_sum(partial_o);
            if (lane == cc) {
                o[out_base + col] = out;
            }
        }
    }

    // Store final state
    const int64_t st_base = (int64_t) bh * K * K;
    #pragma unroll
    for (int rr = 0; rr < rows_per_thread; ++rr) {
        const int row = lane + rr * WARP_SIZE;
        #pragma unroll
        for (int cc = 0; cc < BV; ++cc) {
            const int col = v0 + cc;
            if (col < K) {
                st[st_base + (int64_t) row * K + col] = state[rr][cc];
            }
        }
    }
}

void ggml_cuda_op_gated_delta_rule(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * g    = dst->src[3];
    const ggml_tensor * beta = dst->src[4];
    const ggml_tensor * s    = dst->src[5];

    const int K = (int) q->ne[0];
    const int H = (int) q->ne[1];
    const int T = (int) q->ne[2];
    const int B = (int) q->ne[3];

    const float q_scale = ggml_get_op_params_f32(dst, 0);
    const float eps     = ggml_get_op_params_f32(dst, 1);

    const size_t tsize = ggml_type_size(q->type);
    const int64_t out_elems = (int64_t) K * H * T * B;

    float * dst_d = (float *) dst->data;
    float * o_d   = dst_d;
    float * st_d  = dst_d + out_elems;

    const int64_t q_nb1 = q->nb[1] / tsize;
    const int64_t q_nb2 = q->nb[2] / tsize;
    const int64_t q_nb3 = q->nb[3] / tsize;

    const int64_t k_nb1 = k->nb[1] / tsize;
    const int64_t k_nb2 = k->nb[2] / tsize;
    const int64_t k_nb3 = k->nb[3] / tsize;

    const int64_t v_nb1 = v->nb[1] / tsize;
    const int64_t v_nb2 = v->nb[2] / tsize;
    const int64_t v_nb3 = v->nb[3] / tsize;

    const int64_t g_nb1 = g->nb[1] / tsize;
    const int64_t g_nb2 = g->nb[2] / tsize;

    const int64_t beta_nb1 = beta->nb[1] / tsize;
    const int64_t beta_nb2 = beta->nb[2] / tsize;

    constexpr int BV = 8;
    const dim3 grid((K + BV - 1) / BV, (unsigned) (B * H), 1);
    const dim3 block(WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

    // Use F32 implementation for everything (performing math in float)
    if (q->type == GGML_TYPE_F16) {
        if (s->type == GGML_TYPE_F16) {
            if (K == 64)  gated_delta_rule_fwd<64,  BV, half, half><<<grid, block, 0, stream>>>( (const half *) q->data, (const half *) k->data, (const half *) v->data, (const half *) g->data, (const half *) beta->data, (const half *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else if (K == 128) gated_delta_rule_fwd<128, BV, half, half><<<grid, block, 0, stream>>>( (const half *) q->data, (const half *) k->data, (const half *) v->data, (const half *) g->data, (const half *) beta->data, (const half *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else GGML_ABORT("unsupported head dim");
        } else {
            if (K == 64)  gated_delta_rule_fwd<64,  BV, half, float><<<grid, block, 0, stream>>>( (const half *) q->data, (const half *) k->data, (const half *) v->data, (const half *) g->data, (const half *) beta->data, (const float *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else if (K == 128) gated_delta_rule_fwd<128, BV, half, float><<<grid, block, 0, stream>>>( (const half *) q->data, (const half *) k->data, (const half *) v->data, (const half *) g->data, (const half *) beta->data, (const float *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else GGML_ABORT("unsupported head dim");
        }
    } else {
        if (s->type == GGML_TYPE_F16) {
            if (K == 64)  gated_delta_rule_fwd<64,  BV, float, half><<<grid, block, 0, stream>>>( (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data, (const float *) beta->data, (const half *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else if (K == 128) gated_delta_rule_fwd<128, BV, float, half><<<grid, block, 0, stream>>>( (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data, (const float *) beta->data, (const half *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else GGML_ABORT("unsupported head dim");
        } else {
            if (K == 64)  gated_delta_rule_fwd<64,  BV, float, float><<<grid, block, 0, stream>>>( (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data, (const float *) beta->data, (const float *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else if (K == 128) gated_delta_rule_fwd<128, BV, float, float><<<grid, block, 0, stream>>>( (const float *) q->data, (const float *) k->data, (const float *) v->data, (const float *) g->data, (const float *) beta->data, (const float *) s->data, o_d, st_d, H, T, q_scale, eps, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, g_nb1, g_nb2, beta_nb1, beta_nb2);
            else GGML_ABORT("unsupported head dim");
        }
    }
}
