// kernels/part_66_gradient_clipping.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void grad_norm_kernel(
    float* grad, float* norm,
    int size
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared_mem[tid] = (idx < size) ? grad[idx] * grad[idx] : 0.0f;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        norm[blockIdx.x] = sqrtf(shared_mem[0]);
    }
}

__global__ void grad_clip_kernel(
    float* grad, float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    grad[idx] *= scale;
}