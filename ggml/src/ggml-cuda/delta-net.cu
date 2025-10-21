#include "delta-net.cuh"

// Configure a reasonable block size. We use 256 threads (16x16) for 2D tiling when needed.
#define DELTA_NET_BLOCK_SIZE 16
#define T 256  // Number of threads per block (x-dimension)

#if !defined(LDG)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
#define LDG(ptr) __ldg(ptr)
#else
#define LDG(ptr) (*(ptr))
#endif
#endif

#if !defined(FMA)
#define FMA(a,b,c) fmaf((a),(b),(c))
#endif

#ifndef GGML_DELTA_NET_CHUNK
#define GGML_DELTA_NET_CHUNK 64
#endif

// ============================================================================
// OPTIMIZED DELTA_NET_RECURRENT kernel
// ============================================================================
// Each block processes one (sequence, head) pair
// Token loop is sequential due to state dependency
// 
// PRECISION NOTES: 
// - Uses FMA (fused multiply-add) for all dot products to minimize rounding errors
// - State accumulation is sequential, so precision is maintained through careful ordering
//
// PERFORMANCE OPTIMIZATIONS:
// 1. Vectorized loads (float4) for better memory bandwidth utilization
// 2. Reduced synchronization barriers (6 instead of 7 per token)
// 3. More aggressive loop unrolling for better ILP
// 4. Scalar values kept in registers instead of shared memory
// 5. Better memory access patterns for coalescing
// ============================================================================
__global__ void delta_net_recurrent_f32_kernel(
    const float * __restrict__ q_tokens,      // [n_tokens, S_v, H_v, n_seqs]
    const float * __restrict__ k_tokens,      // [n_tokens, S_v, H_v, n_seqs]
    const float * __restrict__ v_tokens,      // [n_tokens, S_v, H_v, n_seqs]
    const float * __restrict__ g_tokens_exp,  // [n_tokens, 1, H_v, n_seqs]
    const float * __restrict__ beta_tokens,   // [n_tokens, 1, H_v, n_seqs]
    const float * __restrict__ state_in,      // [S_v, S_v, H_v, n_seqs]
    float * __restrict__ output,              // [S_v, H_v, n_tokens, n_seqs]
    float * __restrict__ state_out,           // [S_v, S_v, H_v, n_seqs]
    int64_t S_v,
    int64_t H_v,
    int64_t n_tokens,
    int64_t n_seqs) {
    
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    
    if (head >= H_v || seq >= n_seqs) return;
    
    const int tid = threadIdx.x;

    // Dynamic shared memory: only vectors (scalars in registers)
    extern __shared__ float smem[];
    float * q_vec   = smem;             // S_v
    float * k_vec   = q_vec   + S_v;    // S_v
    float * v_vec   = k_vec   + S_v;    // S_v
    float * kv_mem  = v_vec   + S_v;    // S_v
    float * delta   = kv_mem  + S_v;    // S_v
    float * out_vec = delta   + S_v;    // S_v

    // Offset helper matching CPU layout: [seq][head][i][j]
    const size_t state_base = (size_t)head * (size_t)(S_v * S_v) + (size_t)seq * (size_t)(S_v * S_v * H_v);
    auto off_state = [=](int i, int j) -> size_t {
        return (size_t)j + (size_t)i * S_v + state_base;
    };
    
    auto off_tok_vec = [=](int token, int d) -> size_t {
        return (size_t)token + (size_t)d * n_tokens + (size_t)head * (n_tokens * S_v) + (size_t)seq * (n_tokens * S_v * H_v);
    };
    
    auto off_scalar_tok = [=](const float * base, int token) -> size_t {
        return (size_t)token + (size_t)head * n_tokens + (size_t)seq * (n_tokens * H_v);
    };

    // Initialize state_out with state_in
    const int S_v_sq = S_v * S_v;
    for (int idx = tid; idx < S_v_sq; idx += blockDim.x) {
        int i = idx / S_v;
        int j = idx % S_v;
        state_out[off_state(i, j)] = state_in[off_state(i, j)];
    }
    __syncthreads();
    
    // Process each token sequentially
    for (int token = 0; token < n_tokens; token++) {
        // OPTIMIZATION: Vectorized loads when S_v is aligned to 4
        const bool can_use_vec4 = (S_v % 4 == 0) && ((uintptr_t)&q_tokens[off_tok_vec(token, 0)] % 16 == 0);
        
        if (can_use_vec4) {
            const int vec_count = S_v / 4;
            for (int vec_idx = tid; vec_idx < vec_count; vec_idx += blockDim.x) {
                const int d = vec_idx * 4;
                const size_t base_off = off_tok_vec(token, d);
                
                float4 q4 = *reinterpret_cast<const float4*>(&q_tokens[base_off]);
                float4 k4 = *reinterpret_cast<const float4*>(&k_tokens[base_off]);
                float4 v4 = *reinterpret_cast<const float4*>(&v_tokens[base_off]);
                
                reinterpret_cast<float4*>(&q_vec[d])[0] = q4;
                reinterpret_cast<float4*>(&k_vec[d])[0] = k4;
                reinterpret_cast<float4*>(&v_vec[d])[0] = v4;
            }
        } else {
            // Fallback to scalar loads
            for (int d = tid; d < S_v; d += blockDim.x) {
                q_vec[d] = LDG(&q_tokens[off_tok_vec(token, d)]);
                k_vec[d] = LDG(&k_tokens[off_tok_vec(token, d)]);
                v_vec[d] = LDG(&v_tokens[off_tok_vec(token, d)]);
            }
        }
        
        // Load scalars into shared memory temporarily (broadcast via shuffle would need warp sync)
        __shared__ float scalar_vals[2];
        if (tid == 0) {
            scalar_vals[0] = g_tokens_exp[off_scalar_tok(g_tokens_exp, token)];
            scalar_vals[1] = beta_tokens[off_scalar_tok(beta_tokens, token)];
        }
        __syncthreads();
        float g_exp = scalar_vals[0];
        float beta_val = scalar_vals[1];

        // 1. state = state * g_exp (element-wise multiplication)
        for (int idx = tid; idx < S_v_sq; idx += blockDim.x) {
            int i = idx / S_v;
            int j = idx % S_v;
            state_out[off_state(i, j)] *= g_exp;
        }
        __syncthreads();
        
        // 2. kv_mem[j] = sum_i (state[i,j] * k[i])
        // OPTIMIZATION: More aggressive unrolling
        for (int j = tid; j < S_v; j += blockDim.x) {
            float sum = 0.0f;
            size_t sidx = state_base + (size_t)j;
            
            // Unroll by 8 for better ILP
            int i = 0;
            for (; i + 7 < S_v; i += 8) {
                sum = FMA(state_out[sidx], k_vec[i], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+1], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+2], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+3], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+4], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+5], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+6], sum); sidx += S_v;
                sum = FMA(state_out[sidx], k_vec[i+7], sum); sidx += S_v;
            }
            for (; i < S_v; i++) {
                sum = FMA(state_out[sidx], k_vec[i], sum);
                sidx += S_v;
            }
            kv_mem[j] = sum;
        }
        __syncthreads();
        
        // 3. delta = (v - kv_mem) * beta
        for (int j = tid; j < S_v; j += blockDim.x) {
            delta[j] = (v_vec[j] - kv_mem[j]) * beta_val;
        }
        __syncthreads();
        
        // 4. state[i,j] += k[i] * delta[j]  (outer product)
        for (int idx = tid; idx < S_v_sq; idx += blockDim.x) {
            int i = idx / S_v;
            int j = idx % S_v;
            size_t sidx = state_base + (size_t)j + (size_t)i * (size_t)S_v;
            state_out[sidx] = FMA(k_vec[i], delta[j], state_out[sidx]);
        }
        __syncthreads();
        
        // 5. output[j] = sum_i (state[i,j] * q[i])
        // OPTIMIZATION: Same unrolling strategy
        for (int j = tid; j < S_v; j += blockDim.x) {
            float sum = 0.0f;
            size_t sidx = state_base + (size_t)j;
            
            int i = 0;
            for (; i + 7 < S_v; i += 8) {
                sum = FMA(state_out[sidx], q_vec[i], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+1], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+2], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+3], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+4], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+5], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+6], sum); sidx += S_v;
                sum = FMA(state_out[sidx], q_vec[i+7], sum); sidx += S_v;
            }
            for (; i < S_v; i++) {
                sum = FMA(state_out[sidx], q_vec[i], sum);
                sidx += S_v;
            }
            out_vec[j] = sum;
        }
        __syncthreads();
        
        // Store output for this token
        const size_t output_base = (size_t)head * S_v + (size_t)token * (S_v * H_v) + (size_t)seq * (S_v * H_v * n_tokens);
        if (can_use_vec4) {
            for (int d = tid; d < S_v / 4; d += blockDim.x) {
                *reinterpret_cast<float4*>(&output[output_base + d * 4]) = 
                    *reinterpret_cast<float4*>(&out_vec[d * 4]);
            }
            // Handle remainder
            for (int d = tid + (S_v / 4) * 4; d < S_v; d += blockDim.x) {
                output[output_base + d] = out_vec[d];
            }
        } else {
            for (int d = tid; d < S_v; d += blockDim.x) {
                output[output_base + d] = out_vec[d];
            }
        }
        // OPTIMIZATION: Final sync can be removed if it's the last token
        if (token < n_tokens - 1) {
            __syncthreads();
        }
    }
}

void ggml_cuda_op_delta_net_recurrent(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // q_tokens
    const ggml_tensor * src1 = dst->src[1];  // k_tokens
    const ggml_tensor * src2 = dst->src[2];  // v_tokens
    const ggml_tensor * src3 = dst->src[3];  // g_tokens_exp
    const ggml_tensor * src4 = dst->src[4];  // beta_tokens
    const ggml_tensor * src5 = dst->src[5];  // state
    
    const int64_t H_v = (int64_t) dst->op_params[0];
    const int64_t S_v = (int64_t) dst->op_params[2];
    const int64_t n_tokens = (int64_t) dst->op_params[3];
    const int64_t n_seqs = src0->ne[3];
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    
    // Verify tensor dimensions match CPU expectations
    GGML_ASSERT(src0->ne[3] == n_seqs);  // q tensor
    GGML_ASSERT(src1->ne[3] == n_seqs);  // k tensor
    GGML_ASSERT(src2->ne[3] == n_seqs);  // v tensor
    GGML_ASSERT(src3->ne[3] == n_seqs);  // g tensor
    GGML_ASSERT(src4->ne[3] == n_seqs);  // beta tensor
    GGML_ASSERT(src5->ne[3] == n_seqs);  // state tensor
    
    const float * q_d = (const float *) src0->data;
    const float * k_d = (const float *) src1->data;
    const float * v_d = (const float *) src2->data;
    const float * g_exp_d = (const float *) src3->data;
    const float * beta_d = (const float *) src4->data;
    const float * state_in_d = (const float *) src5->data;
    
    float * dst_d = (float *) dst->data;
    float * output_d = dst_d;
    float * state_out_d = dst_d + (S_v * H_v * n_tokens * n_seqs);
    
    cudaStream_t stream = ctx.stream();
    
    // Launch config
    dim3 grid(H_v, n_seqs);
    int block_x = 256;
    if (S_v < 256) block_x = (S_v >= 128 ? 128 : (S_v >= 64 ? 64 : (S_v >= 32 ? 32 : 16)));
    dim3 block(block_x, 1, 1);
    
    // Shared memory: 6 vectors (S_v each) + 2 scalars for temporary storage
    size_t smem_size = (6 * (size_t)S_v + 2) * sizeof(float);
    
    delta_net_recurrent_f32_kernel<<<grid, block, smem_size, stream>>>(
        q_d, k_d, v_d, g_exp_d, beta_d, state_in_d,
        output_d, state_out_d,
        S_v, H_v, n_tokens, n_seqs
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Chunked kernel for Gated Delta Net
// 
// PRECISION NOTES FOR LONG CONTEXTS (40k+ tokens):
// - g_cumsum values can become large over long sequences. The cumsum operation now uses
//   double precision internally to minimize accumulation errors (see cumsum.cu).
// - State decay uses exp(g_j - g_i) formulation which is numerically stable.
// - FMA (fused multiply-add) is used throughout to minimize rounding errors.
// - For debugging long-context issues: verify g_cumsum precision by comparing with reference.
//
__global__ void delta_net_chunked_f32_kernel(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g_cumsum,
    const float * __restrict__ state_in,
    const float * __restrict__ decay_mask,
    const float * __restrict__ v_beta,
    const float * __restrict__ k_beta,
    const float * __restrict__ attn_in,
    float * __restrict__ output,
    float * __restrict__ state_out,
    float * __restrict__ intermediate_global,  // Global memory for intermediate matrices
    int S_v, int H_v, int n_tokens, int n_seqs, int chunk_size, int num_chunks) {

    const int head = blockIdx.x;
    const int seq  = blockIdx.y;
    const int tid  = threadIdx.x;

    if (head >= H_v || seq >= n_seqs) return;
    
    // Calculate offset for this block's intermediate storage
    const size_t block_idx = (size_t)seq * H_v + head;
    // Each block needs: 4*chunk_size*S_v (value, k_cumdecay, v_prime, v_new) + chunk_size*chunk_size (attn_new)
    const size_t per_block_floats = 4 * (size_t)chunk_size * (size_t)S_v + (size_t)chunk_size * (size_t)chunk_size;
    const size_t intermediate_offset = block_idx * per_block_floats;

    // Offset helpers matching CPU layout
    auto off_qkv = [&](const float * base, int h, int c, int i, int d) -> size_t {
        // dims: [S_v, chunk_size, H_v*num_chunks, n_seqs]
        const int hc = h * num_chunks + c;
        return (size_t)d + (size_t)i * S_v + (size_t)hc * (size_t)(chunk_size * S_v) + (size_t)seq * (size_t)(chunk_size * S_v * H_v * num_chunks);
    };
    auto off_attn = [&](int h, int c, int i, int j) -> size_t {
        // dims: [chunk_size, chunk_size, H_v*num_chunks, n_seqs]
        const int hc = h * num_chunks + c;
        return (size_t)j + (size_t)i * chunk_size + (size_t)hc * (size_t)(chunk_size * chunk_size) + (size_t)seq * (size_t)(chunk_size * chunk_size * H_v * num_chunks);
    };
    auto off_g = [&](int h, int c, int t) -> size_t {
        // dims: [chunk_size, 1, H_v*num_chunks, n_seqs]
        const int hc = h * num_chunks + c;
        return (size_t)t + (size_t)hc * (size_t)chunk_size + (size_t)seq * (size_t)(chunk_size * H_v * num_chunks);
    };
    auto off_state = [&](int i, int j) -> size_t {
        // dims: [S_v, S_v, H_v, n_seqs]
        const size_t state_base = (size_t)head * (size_t)(S_v * S_v) + (size_t)seq * (size_t)(S_v * S_v * H_v);
        return (size_t)j + (size_t)i * S_v + state_base;
    };
    auto off_out = [&](int global_token, int d) -> size_t {
        // dims: [S_v, n_tokens, H_v, n_seqs]
        // CPU layout: d + token * S_v + head * (n_tokens * S_v) + seq * (n_tokens * S_v * H_v)
        return (size_t)d + (size_t)global_token * S_v + (size_t)head * (n_tokens * S_v) + (size_t)seq * (n_tokens * S_v * H_v);
    };

    // Shared memory: only attn_pre + row_buf (small, fits in shared memory)
    extern __shared__ float shmem[];
    float * attn_pre = shmem;
    float * row_buf  = attn_pre + (size_t)chunk_size * chunk_size;
    
    // Global memory pointers for intermediate matrices (avoids shared memory overflow)
    float * value      = intermediate_global + intermediate_offset;
    float * k_cumdecay = value + (size_t)chunk_size * S_v;
    float * v_prime    = k_cumdecay + (size_t)chunk_size * S_v;
    float * v_new      = v_prime + (size_t)chunk_size * S_v;

    // Initialize state_out from state_in
    for (int idx = tid; idx < S_v * S_v; idx += blockDim.x) {
        const int i = idx / S_v;
        const int j = idx % S_v;
        state_out[off_state(i, j)] = state_in[off_state(i, j)];
    }
    __syncthreads();

    // Process each chunk
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int n_tokens_chunk = (chunk == num_chunks - 1 && n_tokens % chunk_size != 0)
            ? (n_tokens % chunk_size)
            : chunk_size;

        // Initialize all attn_pre to zero first
        for (int idx = tid; idx < chunk_size * chunk_size; idx += blockDim.x) {
            attn_pre[idx] = 0.0f;
        }
        __syncthreads();
        
        // Copy attn_in tile to attn_pre (only valid n_tokens_chunk rows/cols)
        for (int idx = tid; idx < n_tokens_chunk * n_tokens_chunk; idx += blockDim.x) {
            int irow = idx / n_tokens_chunk;
            int jcol = idx % n_tokens_chunk;
            attn_pre[irow * chunk_size + jcol] = LDG(&attn_in[off_attn(head, chunk, irow, jcol)]);
        }
        __syncthreads();

        // Triangular updates: for i in 1..n_tokens_chunk-1
        // Python: attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        // where row = attn[..., i, :i] and sub = attn[..., :i, :i]
        // CPU copies row first, then sub, to avoid reading modified values
        for (int irow = 1; irow < n_tokens_chunk; ++irow) {
            // Step 1: Copy row = attn[irow, 0:irow] into row_buf
            for (int k = tid; k < irow; k += blockDim.x) {
                row_buf[k] = attn_pre[irow * chunk_size + k];
            }
            __syncthreads();
            
            // Step 2: Compute new values for attn[irow, 0:irow]
            // The sub matrix attn[:irow, :irow] is read from the CURRENT attn_pre
            // (which contains updates from previous irow iterations)
            for (int j = tid; j < irow; j += blockDim.x) {
                // Compute sum_k (row[k] * sub[k, j]) where k in [0, irow)
                // sub[k, j] = attn_pre[k, j] for k < irow, j < irow
                float sum = 0.0f;
                for (int k = 0; k < irow; ++k) {
                    sum += row_buf[k] * attn_pre[k * chunk_size + j];
                }
                
                // Update: attn[irow, j] = row[j] + sum
                attn_pre[irow * chunk_size + j] = row_buf[j] + sum;
            }
            __syncthreads();
        }
        // Add identity to diagonal
        for (int d = tid; d < n_tokens_chunk; d += blockDim.x) {
            attn_pre[d * chunk_size + d] += 1.0f;
        }
        __syncthreads();

        // ========== OPTIMIZATION: Precompute intermediate matrices in global memory ==========
        // This eliminates massive redundant computation!
        // Note: value, k_cumdecay, v_prime, v_new already declared above using global memory
        
        // Precompute g_exp for all tokens in this chunk and keep it in shared row_buf
        float * g_exp_buf = row_buf;
        for (int t = tid; t < n_tokens_chunk; t += blockDim.x) {
            g_exp_buf[t] = __expf(g_cumsum[off_g(head, chunk, t)]);
        }
        __syncthreads();

        // Compute value = attn_pre @ v_beta [n_tokens_chunk x S_v]
        // OPTIMIZATION: Better loop unrolling and coalescing
        for (int idx = tid; idx < n_tokens_chunk * S_v; idx += blockDim.x) {
            const int row = idx / S_v;
            const int col = idx % S_v;
            float sum = 0.0f;
            const float * __restrict__ pv = &v_beta[off_qkv(v_beta, head, chunk, 0, col)];
            
            // Unroll by 8 for better ILP
            int k = 0;
            for (; k + 7 < n_tokens_chunk; k += 8) {
                sum = FMA(attn_pre[row * chunk_size + k], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 1], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 2], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 3], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 4], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 5], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 6], LDG(pv), sum); pv += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 7], LDG(pv), sum); pv += S_v;
            }
            for (; k < n_tokens_chunk; ++k) {
                sum = FMA(attn_pre[row * chunk_size + k], LDG(pv), sum);
                pv += S_v;
            }
            value[row * S_v + col] = sum;
        }
        __syncthreads();
        
        // Compute k_cumdecay = attn_pre @ (k_beta * exp(g)) [n_tokens_chunk x S_v]
        // OPTIMIZATION: Better unrolling
        for (int idx = tid; idx < n_tokens_chunk * S_v; idx += blockDim.x) {
            const int row = idx / S_v;
            const int col = idx % S_v;
            float sum = 0.0f;
            const float * __restrict__ pk = &k_beta[off_qkv(k_beta, head, chunk, 0, col)];
            
            int k = 0;
            for (; k + 7 < n_tokens_chunk; k += 8) {
                sum = FMA(attn_pre[row * chunk_size + k], LDG(pk) * g_exp_buf[k], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 1], LDG(pk) * g_exp_buf[k + 1], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 2], LDG(pk) * g_exp_buf[k + 2], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 3], LDG(pk) * g_exp_buf[k + 3], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 4], LDG(pk) * g_exp_buf[k + 4], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 5], LDG(pk) * g_exp_buf[k + 5], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 6], LDG(pk) * g_exp_buf[k + 6], sum); pk += S_v;
                sum = FMA(attn_pre[row * chunk_size + k + 7], LDG(pk) * g_exp_buf[k + 7], sum); pk += S_v;
            }
            for (; k < n_tokens_chunk; ++k) {
                sum = FMA(attn_pre[row * chunk_size + k], LDG(pk) * g_exp_buf[k], sum);
                pk += S_v;
            }
            k_cumdecay[row * S_v + col] = sum;
        }
        __syncthreads();
        
        // Compute v_prime = k_cumdecay @ state [n_tokens_chunk x S_v]
        // OPTIMIZATION: Better unrolling for matrix-matrix multiply
        for (int idx = tid; idx < n_tokens_chunk * S_v; idx += blockDim.x) {
            const int row = idx / S_v;
            const int col = idx % S_v;
            float sum = 0.0f;
            const float * __restrict__ pstate_col = &state_out[(size_t)col + (size_t)head * (size_t)(S_v * S_v) + (size_t)seq * (size_t)(S_v * S_v * H_v)];
            
            int k = 0;
            for (; k + 7 < S_v; k += 8) {
                sum = FMA(k_cumdecay[row * S_v + k], pstate_col[k * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 1], pstate_col[(k + 1) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 2], pstate_col[(k + 2) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 3], pstate_col[(k + 3) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 4], pstate_col[(k + 4) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 5], pstate_col[(k + 5) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 6], pstate_col[(k + 6) * S_v], sum);
                sum = FMA(k_cumdecay[row * S_v + k + 7], pstate_col[(k + 7) * S_v], sum);
            }
            for (; k < S_v; ++k) {
                sum = FMA(k_cumdecay[row * S_v + k], pstate_col[k * S_v], sum);
            }
            v_prime[row * S_v + col] = sum;
        }
        __syncthreads();
        
        // Compute v_new = value - v_prime [n_tokens_chunk x S_v]
        for (int idx = tid; idx < n_tokens_chunk * S_v; idx += blockDim.x) {
            v_new[idx] = value[idx] - v_prime[idx];
        }
        __syncthreads();
        
        // ========== OPTIMIZATION 2: Precompute q@k attention matrix ==========
        // Allocate space for attn_new in global memory to avoid recomputing q@k
        float * attn_new = v_new + (size_t)chunk_size * S_v;  // Reuse space after v_new computation
        
        // Compute attn_new = (q @ k.T) * decay_mask [n_tokens_chunk x n_tokens_chunk]
        // Each thread computes multiple elements
        for (int idx = tid; idx < n_tokens_chunk * n_tokens_chunk; idx += blockDim.x) {
            const int i = idx / n_tokens_chunk;
            const int j = idx % n_tokens_chunk;
            
            if (j <= i) {  // Only lower triangular (causal mask)
                float qk_dot = 0.0f;
                const float * __restrict__ pq = &q[off_qkv(q, head, chunk, i, 0)];
                const float * __restrict__ pk = &k[off_qkv(k, head, chunk, j, 0)];
                int d = 0;
                for (; d + 3 < S_v; d += 4) {
                    qk_dot = FMA(LDG(pq + d + 0), LDG(pk + d + 0), qk_dot);
                    qk_dot = FMA(LDG(pq + d + 1), LDG(pk + d + 1), qk_dot);
                    qk_dot = FMA(LDG(pq + d + 2), LDG(pk + d + 2), qk_dot);
                    qk_dot = FMA(LDG(pq + d + 3), LDG(pk + d + 3), qk_dot);
                }
                for (; d < S_v; ++d) {
                    qk_dot = FMA(LDG(pq + d), LDG(pk + d), qk_dot);
                }
                attn_new[i * chunk_size + j] = qk_dot * LDG(&decay_mask[off_attn(head, chunk, i, j)]);
            } else {
                attn_new[i * chunk_size + j] = 0.0f;  // Upper triangular is zero
            }
        }
        __syncthreads();
        
        
        // ========== Now compute output using PRECOMPUTED matrices ==========
        for (int idx = tid; idx < n_tokens_chunk * S_v; idx += blockDim.x) {
            const int row = idx / S_v;
            const int col = idx % S_v;
            
            // attn_inter = (q * exp(g)) @ state - use precomputed g_exp
            float attn_inter = 0.0f;
            const float g_exp = g_exp_buf[row];
            const float * __restrict__ pqrow = &q[off_qkv(q, head, chunk, row, 0)];
            const float * __restrict__ pstate_col = &state_out[(size_t)col + (size_t)head * (size_t)(S_v * S_v) + (size_t)seq * (size_t)(S_v * S_v * H_v)];
            #pragma unroll 4
            for (int k_idx = 0; k_idx < S_v; ++k_idx) {
                attn_inter = FMA(LDG(pqrow + k_idx) * g_exp, pstate_col[(size_t)k_idx * (size_t)S_v], attn_inter);
            }
            
            // core_attn_out = attn_new @ v_new using PRECOMPUTED attn_new and v_new!
            float core_attn_out = 0.0f;
            for (int k_idx = 0; k_idx <= row; ++k_idx) {
                // Use precomputed attn_new - NO q@k computation!
                core_attn_out += attn_new[row * chunk_size + k_idx] * v_new[k_idx * S_v + col];
            }
            
            const int global_token = chunk * chunk_size + row;
            if (global_token < n_tokens) {
                output[off_out(global_token, col)] = attn_inter + core_attn_out;
            }
        }
        __syncthreads();

        // ========== Update state using PRECOMPUTED v_new ==========
        // Precompute g_diff_exp values (reuse g_exp_buf)
        float g_last = g_exp_buf[n_tokens_chunk - 1];
        float * g_diff_buf = g_exp_buf;  // Reuse buffer
        // Use exp of the difference to avoid divide-by-zero/underflow issues
        const float g_last_log = g_cumsum[off_g(head, chunk, n_tokens_chunk - 1)];
        for (int t = tid; t < n_tokens_chunk; t += blockDim.x) {
            g_diff_buf[t] = __expf(g_last_log - g_cumsum[off_g(head, chunk, t)]);
        }
        __syncthreads();
        
        for (int idx = tid; idx < S_v * S_v; idx += blockDim.x) {
            const int i = idx / S_v;
            const int j = idx % S_v;
            
            float new_state_val = state_out[off_state(i, j)] * g_last;
            
            // Use precomputed v_new and g_diff - NO exp() calls in loop!
            const float * __restrict__ pk_tok = &k[off_qkv(k, head, chunk, 0, i)];
            #pragma unroll 4
            for (int t = 0; t < n_tokens_chunk; ++t) {
                new_state_val = FMA(LDG(pk_tok), g_diff_buf[t] * v_new[t * S_v + j], new_state_val);
                pk_tok += (size_t)S_v;
            }
            
            state_out[off_state(i, j)] = new_state_val;
        }
        __syncthreads();
    }
}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // inputs
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g (cumsum)
    const ggml_tensor * src4 = dst->src[4];  // state
    const ggml_tensor * src5 = dst->src[5];  // decay_mask
    const ggml_tensor * src6 = dst->src[6];  // v_beta
    const ggml_tensor * src7 = dst->src[7];  // k_beta
    const ggml_tensor * src8 = dst->src[8];  // attn (pre)

    const int H_v       = (int) dst->op_params[0];
    const int S_v       = (int) dst->op_params[2];
    const int n_tokens  = (int) dst->op_params[3];
    const int n_seqs    = (int) src0->ne[3];
    const int chunk_size = (int) GGML_DELTA_NET_CHUNK;
    const int pad_size   = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int num_chunks = (n_tokens + pad_size) / chunk_size;

    const float * q_d   = (const float *) src0->data;
    const float * k_d   = (const float *) src1->data;
    const float * v_d   = (const float *) src2->data;
    const float * g_d   = (const float *) src3->data;
    const float * state_in_d = (const float *) src4->data;
    const float * decay_d = (const float *) src5->data;
    const float * vbeta_d = (const float *) src6->data;
    const float * kbeta_d = (const float *) src7->data;
    const float * attn_in_d = (const float *) src8->data;

    float * dst_d = (float *) dst->data;
    float * output_d   = dst_d;
    float * state_out_d = dst_d + (size_t) S_v * H_v * n_tokens * n_seqs;

    dim3 grid(H_v, n_seqs);
    int block_x2 = 256;
    if (S_v < 256) block_x2 = (S_v >= 128 ? 128 : (S_v >= 64 ? 64 : (S_v >= 32 ? 32 : 16)));
    dim3 block(block_x2, 1, 1);

    cudaStream_t stream = ctx.stream();
    
    // Allocate global memory for intermediate matrices per block:
    // - value, k_cumdecay, v_prime, v_new: 4 * chunk_size * S_v
    // - attn_new: chunk_size * chunk_size (reuses space after v_new)
    // Total: max(4 * chunk_size * S_v, 4 * chunk_size * S_v + chunk_size * chunk_size)
    size_t intermediate_size = (4 * (size_t)chunk_size * S_v + (size_t)chunk_size * chunk_size) * sizeof(float);
    ggml_cuda_pool_alloc<float> intermediate_alloc(ctx.pool(), intermediate_size * H_v * n_seqs);
    float * intermediate_d = intermediate_alloc.get();
    
    // Shared memory per block: only attn_pre + row_buf (much smaller!)
    size_t smem = ((size_t)chunk_size * chunk_size + chunk_size) * sizeof(float);
    
    delta_net_chunked_f32_kernel<<<grid, block, smem, stream>>>(
        q_d, k_d, v_d, g_d, state_in_d, decay_d, vbeta_d, kbeta_d, attn_in_d,
        output_d, state_out_d, intermediate_d,
        S_v, H_v, n_tokens, n_seqs, chunk_size, num_chunks
    );
    
    CUDA_CHECK(cudaGetLastError());
}

