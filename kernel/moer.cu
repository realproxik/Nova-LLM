// kernels/part_59_moe_router.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_router_kernel(
    float* input, float* router_weights, int* expert_indices, float* expert_weights,
    int batch, int seq, int hidden, int num_experts, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq;
    
    if (idx >= total) return;
    
    int b = idx / seq;
    int s = idx % seq;
    
    float* in = input + (b * seq + s) * hidden;
    
    float scores[8];
    int indices[8];
    
    for (int e = 0; e < num_experts; e++) {
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum += in[i] * router_weights[e * hidden + i];
        }
        scores[e] = sum;
        indices[e] = e;
    }
    
    for (int i = 0; i < num_experts - 1; i++) {
        for (int j = i + 1; j < num_experts; j++) {
            if (scores[i] < scores[j]) {
                float temp_s = scores[i];
                scores[i] = scores[j];
                scores[j] = temp_s;
                int temp_i = indices[i];
                indices[i] = indices[j];
                indices[j] = temp_i;
            }
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        scores[i] = expf(scores[i]);
        sum_exp += scores[i];
    }
    
    for (int i = 0; i < top_k; i++) {
        expert_indices[(b * seq + s) * top_k + i] = indices[i];
        expert_weights[(b * seq + s) * top_k + i] = scores[i] / sum_exp;
    }
}