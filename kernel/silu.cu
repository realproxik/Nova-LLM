// kernels/part_16_silu.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void silu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

__global__ void silu_fast_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x * (x > 0.0f ? 1.0f : 0.0f);
}