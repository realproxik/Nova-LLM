// kernels/part_17_relu.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x > 0.0f ? x : 0.0f;
}

__global__ void relu6_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x > 0.0f ? (x < 6.0f ? x : 6.0f) : 0.0f;
}

__global__ void leaky_relu_kernel(float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * x;
}