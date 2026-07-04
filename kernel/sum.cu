// kernels/part_31_reduction_sum.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce_sum_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

__global__ void reduce_sum_warps_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    float sum = (idx < size) ? input[idx] : 0.0f;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) {
        shared_mem[warp] = sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? shared_mem[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}