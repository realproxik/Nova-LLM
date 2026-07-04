// kernels/part_29_cast_ops.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cast_fp32_to_int8_kernel(float* input, int8_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = input[idx];
    val = fminf(127.0f, fmaxf(-127.0f, val));
    output[idx] = (int8_t)roundf(val);
}

__global__ void cast_int8_to_fp32_kernel(int8_t* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = (float)input[idx];
}

__global__ void cast_fp32_to_int32_kernel(float* input, int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = (int)roundf(input[idx]);
}

__global__ void cast_int32_to_fp32_kernel(int* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = (float)input[idx];
}