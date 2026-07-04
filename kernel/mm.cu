// kernels/part_32_reduction_max_min.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce_max_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (idx < size) ? input[idx] : -INFINITY;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

__global__ void reduce_min_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (idx < size) ? input[idx] : INFINITY;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fminf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}