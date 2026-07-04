// kernels/part_34_scatter_gather.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gather_kernel(float* input, int* indices, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[indices[idx]];
}

__global__ void scatter_kernel(float* input, int* indices, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[indices[idx]] = input[idx];
}

__global__ void gather_nd_kernel(float* input, int* indices, float* output, int batch, int dims, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int offset = 0;
    for (int i = 0; i < dims; i++) {
        offset = offset * batch + indices[idx * dims + i];
    }
    output[idx] = input[offset];
}