#include "common.cuh"

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_delta_net_recurrent(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

