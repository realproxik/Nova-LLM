// kernels/part_119_fused_reglu.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_reglu_kernel(
    float* input, float* output,
    float* w1, float* w2, float* w3,
    int batch, int seq, int hidden, int intermediate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * hidden;
    
    if (idx >= total) return;
    
    int b = (idx / hidden) / seq;
    int s = (idx / hidden) % seq;
    int h = idx % hidden;
    int token_idx = b * seq * hidden + s * hidden;
    
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < hidden; i++) {
        sum1 += input[token_idx + i] * w1[i * hidden + h];
        sum2 += input[token_idx + i] * w3[i * hidden + h];
    }
    
    float reglu = (sum1 > 0 ? sum1 : 0) * sum2;
    output[idx] = reglu;
}