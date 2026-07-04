// kernels/part_103_fused_mlp_residual.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_mlp_residual_kernel(
    float* input, float* output,
    float* w1, float* w2,
    int batch, int seq, int hidden, int intermediate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq * hidden + s * hidden;
    
    float temp[4096 * 4];
    for (int i = 0; i < intermediate; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += input[token_idx + j] * w1[i * hidden + j];
        }
        temp[i] = sum > 0 ? sum : 0;
    }
    
    float mlp_out = 0.0f;
    for (int i = 0; i < intermediate; i++) {
        mlp_out += temp[i] * w2[h * intermediate + i];
    }
    
    output[idx] = mlp_out + input[token_idx + h];
}