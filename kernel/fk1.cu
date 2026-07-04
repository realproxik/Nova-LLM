// kernels/part_100_fused_qkv_rope.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_qkv_rope_kernel(
    float* input, float* q, float* k, float* v,
    float* q_weight, float* k_weight, float* v_weight,
    int batch, int seq, int hidden, int dim,
    float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq * hidden + s * hidden;
    int base_idx = b * seq * dim + s * dim;
    
    float q_val = 0.0f;
    float k_val = 0.0f;
    float v_val = 0.0f;
    
    for (int i = 0; i < hidden; i++) {
        q_val += input[token_idx + i] * q_weight[i * hidden + h];
        k_val += input[token_idx + i] * k_weight[i * hidden + h];
        v_val += input[token_idx + i] * v_weight[i * hidden + h];
    }
    
    int head_dim = dim / 32;
    int head = h / head_dim;
    int head_offset = h % head_dim;
    
    if (head_offset < head_dim / 2) {
        float angle = (float)s / powf(theta, (float)head_offset / head_dim);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float q_shift = q_val;
        float k_shift = k_val;
        
        int idx_pair = base_idx + head * head_dim + head_offset + head_dim / 2;
        q[base_idx + h] = q_val * cos_val - q[idx_pair] * sin_val;
        k[base_idx + h] = k_val * cos_val - k[idx_pair] * sin_val;
        q[idx_pair] = q_val * sin_val + q_shift * cos_val;
        k[idx_pair] = k_val * sin_val + k_shift * cos_val;
    }
    
    v[base_idx + h] = v_val;
}