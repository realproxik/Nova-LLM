// kernels/part_27_fp16_fp32_convert.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fp16_to_fp32_kernel(half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = __half2float(input[idx]);
}

__global__ void fp32_to_fp16_kernel(float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = __float2half(input[idx]);
}

__global__ void fp16_to_fp32_vectorized_kernel(half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    half2 val2 = ((half2*)input)[idx];
    float2 valf2 = __half22float2(val2);
    ((float2*)output)[idx] = valf2;
}