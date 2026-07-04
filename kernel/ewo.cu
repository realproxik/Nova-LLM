// kernels/part_30_elementwise_ops.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = a[idx] + b[idx];
}

__global__ void sub_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = a[idx] - b[idx];
}

__global__ void mul_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = a[idx] * b[idx];
}

__global__ void div_kernel(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = a[idx] / (b[idx] + 1e-7f);
}

__global__ void pow_kernel(float* a, float power, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = powf(a[idx], power);
}

__global__ void sqrt_kernel(float* a, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = sqrtf(fmaxf(a[idx], 0.0f));
}

__global__ void exp_kernel(float* a, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = expf(a[idx]);
}

__global__ void log_kernel(float* a, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    c[idx] = logf(fmaxf(a[idx], 1e-7f));
}