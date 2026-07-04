// kernels/part_15_gelu.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = 0.5f * x * (1.0f + erf(x / sqrtf(2.0f)));
}

__global__ void gelu_approx_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    output[idx] = 0.5f * x * (1.0f + tanh(sqrtf(2.0f / M_PI) * (x + 0.044715f * x3)));
}

__global__ void gelu_fast_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-1.702f * x));
}