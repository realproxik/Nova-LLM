// kernels/part_104_fused_dropout_residual.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_dropout_residual_kernel(
    float* input, float* residual, float* output,
    float* mask, float rate,
    int size, unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    unsigned int state = seed + idx * 1103515245 + 12345;
    float random = (float)(state & 0x7fffffff) / (float)0x7fffffff;
    
    float keep = 1.0f - rate;
    if (random < keep) {
        float val = input[idx] / keep;
        output[idx] = val + residual[idx];
        mask[idx] = 1.0f;
    } else {
        output[idx] = residual[idx];
        mask[idx] = 0.0f;
    }
}