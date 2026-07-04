// kernels/part_89_moe_alltoall_dispatch.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_alltoall_dispatch_kernel(
    float* input, float* output,
    int* expert_indices, int* expert_counts,
    int batch, int seq, int hidden,
    int num_experts, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * seq) return;
    
    int b = idx / seq;
    int s = idx % seq;
    int token_idx = b * seq + s;
    
    for (int k = 0; k < top_k; k++) {
        int expert = expert_indices[token_idx * top_k + k];
        int offset = atomicAdd(&expert_counts[expert], hidden);
        for (int i = 0; i < hidden; i++) {
            output[expert * batch * seq * hidden + offset + i] =
                input[token_idx * hidden + i];
        }
    }
}