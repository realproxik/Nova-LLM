// kernels/part_33_scan.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scan_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float val = (tid >= stride) ? shared_mem[tid - stride] : 0.0f;
        __syncthreads();
        if (tid >= stride) {
            shared_mem[tid] += val;
        }
        __syncthreads();
    }
    
    if (idx < size) {
        output[idx] = shared_mem[tid];
    }
}

__global__ void scan_warps_kernel(float* input, float* output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    float val = (idx < size) ? input[idx] : 0.0f;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    
    if (lane == 31) {
        shared_mem[warp] = val;
    }
    __syncthreads();
    
    if (lane < 32) {
        float warp_sum = (warp < blockDim.x / 32) ? shared_mem[warp] : 0.0f;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float n = __shfl_up_sync(0xffffffff, warp_sum, offset);
            if (lane >= offset) warp_sum += n;
        }
        
        if (tid < 32) {
            shared_mem[tid] = warp_sum;
        }
    }
    __syncthreads();
    
    if (tid < blockDim.x / 32) {
        float warp_sum = shared_mem[tid];
        val += warp_sum;
    }
    
    if (idx < size) {
        output[idx] = val;
    }
}