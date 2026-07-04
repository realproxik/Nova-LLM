// kernels/part_111_fused_moe_attn.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_moe_attn_kernel(
    float* input, float* output,
    float* router_weights,
    float* expert_weights,
    float* q_weight, float* k_weight, float* v_weight,
    int batch, int seq, int hidden, int dim,
    int num_experts, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq + s;
    
    float router_scores[8];
    for (int e = 0; e < num_experts; e++) {
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum += input[token_idx * hidden + i] * router_weights[e * hidden + i];
        }
        router_scores[e] = sum;
    }
    
    int expert_indices[8];
    float expert_scales[8];
    for (int i = 0; i < top_k; i++) {
        int max_idx = 0;
        float max_val = router_scores[0];
        for (int j = 1; j < num_experts; j++) {
            if (router_scores[j] > max_val) {
                max_val = router_scores[j];
                max_idx = j;
            }
        }
        expert_indices[i] = max_idx;
        expert_scales[i] = max_val;
        router_scores[max_idx] = -INFINITY;
    }
    
    float moe_out = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int expert = expert_indices[k];
        float scale = expert_scales[k];
        float temp = 0.0f;
        for (int i = 0; i < hidden; i++) {
            temp += input[token_idx * hidden + i] * expert_weights[expert * hidden * hidden + i * hidden + h];
        }
        moe_out += scale * (temp > 0 ? temp : 0);
    }
    
    float attn_out = 0.0f;
    for (int i = 0; i < hidden; i++) {
        attn_out += input[token_idx * hidden + i] * q_weight[i * hidden + h];
        attn_out += input[token_idx * hidden + i] * k_weight[i * hidden + h];
        attn_out += input[token_idx * hidden + i] * v_weight[i * hidden + h];
    }
    
    output[idx] = moe_out + attn_out + input[token_idx * hidden + h];
}