// kernels/part_101_fused_attn_mlp.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_attn_mlp_kernel(
    float* input, float* output,
    float* q_weight, float* k_weight, float* v_weight, float* o_weight,
    float* mlp_w1, float* mlp_w2,
    int batch, int seq, int hidden, int dim,
    int intermediate
) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq * hidden + s * hidden;
    
    // Attention part
    float attn_out = 0.0f;
    for (int i = 0; i < hidden; i++) {
        attn_out += input[token_idx + i] * o_weight[i * hidden + h];
    }
    
    // MLP part
    float temp[4096 * 4];
    for (int i = 0; i < intermediate; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += input[token_idx + j] * mlp_w1[i * hidden + j];
        }
        temp[i] = sum > 0 ? sum : 0;
    }
    
    float mlp_out = 0.0f;
    for (int i = 0; i < intermediate; i++) {
        mlp_out += temp[i] * mlp_w2[h * intermediate + i];
    }
    
    output[idx] = attn_out + mlp_out + input[token_idx + h];
}