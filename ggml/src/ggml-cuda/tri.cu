#include "tri.cuh"

// Optimized: process 4 elements per thread with float4
template<ggml_tri_type type>
__global__ void tri_f32_kernel(const float * __restrict__ x, float * __restrict__ dst,
                                int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
                                int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3,
                                float constant, bool keep_org_val) {
    
    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;
    
    if (i03 >= ne3 || i02 >= ne2 || i01 >= ne1) return;
    
    const float * src_row = (const float *)((const char *)x + i01*nb1 + i02*nb2 + i03*nb3);
    float * dst_row = (float *)((char *)dst + i01*dst_nb1 + i02*dst_nb2 + i03*dst_nb3);
    
    const int row = i01;
    const int tid = threadIdx.x;
    
    // Vectorized: process 4 elements at once when possible
    const int64_t vec_count = ne0 / 4;
    const int64_t remainder = ne0 % 4;
    
    // Process 4 elements at a time
    for (int64_t vec_idx = tid; vec_idx < vec_count; vec_idx += blockDim.x) {
        const int64_t col_base = vec_idx * 4;
        
        // Load 4 values
        float4 src_val = *reinterpret_cast<const float4*>(&src_row[col_base]);
        float4 dst_val;
        
        // Process each element
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = col_base + i;
            bool cmp;
            switch (type) {
                case GGML_TRI_TYPE_LOWER:      cmp = col < row;  break;
                case GGML_TRI_TYPE_LOWER_DIAG: cmp = col <= row; break;
                case GGML_TRI_TYPE_UPPER:      cmp = col > row;  break;
                case GGML_TRI_TYPE_UPPER_DIAG: cmp = col >= row; break;
                default: cmp = false; break;
            }
            (&dst_val.x)[i] = cmp ? (keep_org_val ? (&src_val.x)[i] : constant) : 0.0f;
        }
        
        // Store 4 values
        *reinterpret_cast<float4*>(&dst_row[col_base]) = dst_val;
    }
    
    // Handle remainder elements
    for (int64_t i = tid + vec_count * 4; i < ne0; i += blockDim.x) {
        const int col = i;
        bool cmp;
        switch (type) {
            case GGML_TRI_TYPE_LOWER:      cmp = col < row;  break;
            case GGML_TRI_TYPE_LOWER_DIAG: cmp = col <= row; break;
            case GGML_TRI_TYPE_UPPER:      cmp = col > row;  break;
            case GGML_TRI_TYPE_UPPER_DIAG: cmp = col >= row; break;
            default: cmp = false; break;
        }
        dst_row[col] = cmp ? (keep_org_val ? src_row[col] : constant) : 0.0f;
    }
}

void ggml_cuda_op_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[0] == src0->ne[1]); // Square matrices
    
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    
    cudaStream_t stream = ctx.stream();
    
    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];
    
    const int64_t nb0 = src0->nb[0];
    const int64_t nb1 = src0->nb[1];
    const int64_t nb2 = src0->nb[2];
    const int64_t nb3 = src0->nb[3];
    
    const int64_t dst_nb0 = dst->nb[0];
    const int64_t dst_nb1 = dst->nb[1];
    const int64_t dst_nb2 = dst->nb[2];
    const int64_t dst_nb3 = dst->nb[3];
    
    ggml_tri_type ttype = (ggml_tri_type) dst->op_params[0];
    float constant = ggml_get_op_params_f32(dst, 1);
    bool keep_org_val = isnan(constant);
    
    // Launch kernel
    // Grid: (ne1 rows, ne2 batches, ne3 batches)
    // Block: (ne0 cols per row)
    dim3 grid(ne1, ne2, ne3);
    int block_size = min((int)ne0, 1024);
    
    // We need to launch multiple blocks per row if ne0 > 1024
    int num_blocks_per_row = (ne0 + block_size - 1) / block_size;
    
    if (num_blocks_per_row > 1) {
        // For very wide matrices, use 2D grid with multiple blocks per row
        dim3 grid_2d(ne1 * num_blocks_per_row, ne2, ne3);
        
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:
                tri_f32_kernel<GGML_TRI_TYPE_LOWER><<<grid_2d, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_LOWER_DIAG:
                tri_f32_kernel<GGML_TRI_TYPE_LOWER_DIAG><<<grid_2d, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_UPPER:
                tri_f32_kernel<GGML_TRI_TYPE_UPPER><<<grid_2d, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_UPPER_DIAG:
                tri_f32_kernel<GGML_TRI_TYPE_UPPER_DIAG><<<grid_2d, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
        }
    } else {
        // Standard case: one block per row
        switch (ttype) {
            case GGML_TRI_TYPE_LOWER:
                tri_f32_kernel<GGML_TRI_TYPE_LOWER><<<grid, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_LOWER_DIAG:
                tri_f32_kernel<GGML_TRI_TYPE_LOWER_DIAG><<<grid, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_UPPER:
                tri_f32_kernel<GGML_TRI_TYPE_UPPER><<<grid, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
            case GGML_TRI_TYPE_UPPER_DIAG:
                tri_f32_kernel<GGML_TRI_TYPE_UPPER_DIAG><<<grid, block_size, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3, constant, keep_org_val);
                break;
        }
    }
}

