// kernels/part_25_dequant_int8.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dequantize_int8_kernel(int8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = (float)input[idx] / scale;
}

__global__ void dequantize_int8_per_tensor_kernel(int8_t* input, float* output, float* scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int bid = blockIdx.x;
    output[idx] = (float)input[idx] / scale[bid];
}