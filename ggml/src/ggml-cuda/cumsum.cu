#include "cumsum.cuh"

// Warp-level inclusive scan (cumulative sum)
template<int WARP_SZ>
__device__ inline float warp_cumsum(float val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SZ; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += n;
    }
    return val;
}

// Kernel for small rows (row_len <= 1024)
// Each block processes one row
template<int BLOCK_SIZE>
__global__ void cumsum_f32_kernel(const float * __restrict__ x, float * __restrict__ dst, 
                                   int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                   int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
                                   int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3) {
    
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    
    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;
    
    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / 32;
    const int num_warps = BLOCK_SIZE / 32;
    
    __shared__ float warp_sums[32]; // max 32 warps per block
    
    // Use register for carry instead of shared memory - faster!
    float carry_accum = 0.0f;
    
    // Process elements in chunks of BLOCK_SIZE
    for (int64_t i = tid; i < ne0; i += BLOCK_SIZE) {
        float val = src_row[i];
        
        // Warp-level scan
        float warp_sum = warp_cumsum<32>(val, lane_id);
        
        // Get the total sum from this warp and broadcast to all warps
        if (lane_id == 31) {
            warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();
        
        // Thread 0 computes prefix sum of warp totals
        __shared__ float tile_carry;
        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < num_warps; w++) {
                float tmp = warp_sums[w];
                warp_sums[w] = s; // warp prefix offset within this tile
                s += tmp;         // accumulate total of this tile
            }
            tile_carry = carry_accum; // carry to add to this tile's results
            carry_accum += s;         // update carry for next tile (register!)
        }
        __syncthreads();

        // Add warp prefix and previous tile carry
        float result = warp_sum + warp_sums[warp_id] + tile_carry;
        dst_row[i] = result;
    }
}

// Fallback for very large rows: sequential processing
__global__ void cumsum_f32_sequential_kernel(const float * __restrict__ x, float * __restrict__ dst,
                                              int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                              int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
                                              int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3) {
    
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    
    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;
    if (threadIdx.x != 0) return; // Only first thread in block
    
    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);
    
    float cumsum = 0.0f;
    for (int64_t i = 0; i < ne0; i++) {
        cumsum += src_row[i];
        dst_row[i] = cumsum;
    }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    
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
    
    // Launch kernel
    dim3 grid(ne1, ne2, ne3);
    
    if (ne0 <= 4096) {
        // Use parallel scan for small to medium rows
        const int block_size = 256;
        cumsum_f32_kernel<block_size><<<grid, block_size, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    } else {
        // Use sequential kernel for very large rows
        cumsum_f32_sequential_kernel<<<grid, 1, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    }
}

