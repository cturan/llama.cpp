#include "cumsum.cuh"

// Warp-level inclusive scan (cumulative sum)
// Uses double precision internally for better accuracy at long sequences
// Note: CUDA doesn't have native double shuffle, so we use int64 reinterpretation
// OPTIMIZATION: Fully unrolled for WARP_SZ=32 for better performance
template<int WARP_SZ>
__device__ __forceinline__ double warp_cumsum_double(double val, int lane_id) {
    static_assert(WARP_SZ == 32, "Only warp size 32 is supported");
    
    // Fully unrolled loop for maximum performance
    long long int val_as_int, n_as_int;
    double n;
    
    // offset = 1
    val_as_int = __double_as_longlong(val);
    n_as_int = __shfl_up_sync(0xffffffff, val_as_int, 1);
    n = __longlong_as_double(n_as_int);
    if (lane_id >= 1) val += n;
    
    // offset = 2
    val_as_int = __double_as_longlong(val);
    n_as_int = __shfl_up_sync(0xffffffff, val_as_int, 2);
    n = __longlong_as_double(n_as_int);
    if (lane_id >= 2) val += n;
    
    // offset = 4
    val_as_int = __double_as_longlong(val);
    n_as_int = __shfl_up_sync(0xffffffff, val_as_int, 4);
    n = __longlong_as_double(n_as_int);
    if (lane_id >= 4) val += n;
    
    // offset = 8
    val_as_int = __double_as_longlong(val);
    n_as_int = __shfl_up_sync(0xffffffff, val_as_int, 8);
    n = __longlong_as_double(n_as_int);
    if (lane_id >= 8) val += n;
    
    // offset = 16
    val_as_int = __double_as_longlong(val);
    n_as_int = __shfl_up_sync(0xffffffff, val_as_int, 16);
    n = __longlong_as_double(n_as_int);
    if (lane_id >= 16) val += n;
    
    return val;
}

// Original float version for backward compatibility
template<int WARP_SZ>
__device__ __forceinline__ float warp_cumsum(float val, int lane_id) {
    // Use double precision internally, cast back to float
    return (float)warp_cumsum_double<WARP_SZ>((double)val, lane_id);
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
    
    __shared__ double warp_sums[32]; // max 32 warps per block - use double for precision
    
    // Use double precision for carry to prevent precision loss at long sequences (40k+ tokens)
    double carry_accum = 0.0;
    
    // Process elements in chunks of BLOCK_SIZE
    for (int64_t i = tid; i < ne0; i += BLOCK_SIZE) {
        float val = src_row[i];
        
        // Warp-level scan using double precision internally
        double warp_sum = warp_cumsum_double<32>((double)val, lane_id);
        
        // Get the total sum from this warp and broadcast to all warps
        if (lane_id == 31) {
            warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();
        
        // Thread 0 computes prefix sum of warp totals
        __shared__ double tile_carry;
        if (tid == 0) {
            double s = 0.0;
            for (int w = 0; w < num_warps; w++) {
                double tmp = warp_sums[w];
                warp_sums[w] = s; // warp prefix offset within this tile
                s += tmp;         // accumulate total of this tile
            }
            tile_carry = carry_accum; // carry to add to this tile's results
            carry_accum += s;         // update carry for next tile (register!)
        }
        __syncthreads();

        // Add warp prefix and previous tile carry, then cast to float for output
        double result = warp_sum + warp_sums[warp_id] + tile_carry;
        dst_row[i] = (float)result;
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
    
    // Use double precision for accumulator to prevent precision loss at long sequences
    double cumsum = 0.0;
    for (int64_t i = 0; i < ne0; i++) {
        cumsum += (double)src_row[i];
        dst_row[i] = (float)cumsum;
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
    
    // OPTIMIZATION: Dynamic block size selection for better occupancy
    if (ne0 <= 8192) {
        // Use parallel scan for small to medium rows
        // Choose block size based on row length for better occupancy
        if (ne0 <= 256) {
            cumsum_f32_kernel<128><<<grid, 128, 0, stream>>>(
                src0_d, dst_d, ne0, ne1, ne2, ne3,
                nb0, nb1, nb2, nb3,
                dst_nb0, dst_nb1, dst_nb2, dst_nb3
            );
        } else if (ne0 <= 2048) {
            cumsum_f32_kernel<256><<<grid, 256, 0, stream>>>(
                src0_d, dst_d, ne0, ne1, ne2, ne3,
                nb0, nb1, nb2, nb3,
                dst_nb0, dst_nb1, dst_nb2, dst_nb3
            );
        } else {
            cumsum_f32_kernel<512><<<grid, 512, 0, stream>>>(
                src0_d, dst_d, ne0, ne1, ne2, ne3,
                nb0, nb1, nb2, nb3,
                dst_nb0, dst_nb1, dst_nb2, dst_nb3
            );
        }
    } else {
        // Use sequential kernel for very large rows
        cumsum_f32_sequential_kernel<<<grid, 1, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    }
}

