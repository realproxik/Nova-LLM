// kernels/part_95_expert_parallel_combine.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void expert_parallel_combine_kernel(
    float* expert_output, float* output,
    int* expert_assignment, float* expert_scores,
    int* expert_offsets,
    int batch, int seq, int hidden,
    int num_experts, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * seq * hidden) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq + s;
    
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int expert = expert_assignment[token_idx * top_k + k];
        float score = expert_scores[token_idx * top_k + k];
        int offset = expert_offsets[expert] + token_idx * hidden + h;
        sum += score * expert_output[expert * batch * seq * hidden + offset];
    }
    output[idx] = sum;
}