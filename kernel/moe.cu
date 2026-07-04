// kernels/part_60_moe_expert_forward.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_expert_forward_kernel(
    float* input, float* output,
    float* expert_weights,
    int* expert_indices,
    float* expert_scores,
    int batch, int seq, int hidden, int intermediate,
    int num_experts, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    
    int token_idx = b * seq + s;
    float sum = 0.0f;
    
    for (int k = 0; k < top_k; k++) {
        int expert = expert_indices[token_idx * top_k + k];
        float score = expert_scores[token_idx * top_k + k];
        
        float* in = input + token_idx * hidden;
        float* w1 = expert_weights + expert * (hidden * intermediate + intermediate * hidden);
        float* w2 = w1 + hidden * intermediate;
        
        float temp[4096 * 4];
        for (int i = 0; i < intermediate; i++) {
            float val = 0.0f;
            for (int j = 0; j < hidden; j++) {
                val += in[j] * w1[i * hidden + j];
            }
            temp[i] = val > 0 ? val : 0;
        }
        
        float val = 0.0f;
        for (int i = 0; i < intermediate; i++) {
            val += temp[i] * w2[h * intermediate + i];
        }
        sum += score * val;
    }
    
    output[idx] = sum;
}