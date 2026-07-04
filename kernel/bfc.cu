// kernels/part_28_bf16_fp32_convert.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void bf16_to_fp32_kernel(__nv_bfloat16* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = __bfloat162float(input[idx]);
}

__global__ void fp32_to_bf16_kernel(float* input, __nv_bfloat16* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = __float2bfloat16(input[idx]);
}