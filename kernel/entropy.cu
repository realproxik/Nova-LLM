// kernels/part_19_cross_entropy.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(float* logits, int* targets, float* loss, int batch, int vocab) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid;
    
    shared_mem[tid] = 0.0f;
    
    float max_val = logits[b * vocab];
    for (int i = 1; i < vocab; i++) {
        if (logits[b * vocab + i] > max_val) max_val = logits[b * vocab + i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        sum += expf(logits[b * vocab + i] - max_val);
    }
    
    int target = targets[b];
    float prob = expf(logits[b * vocab + target] - max_val) / sum;
    
    shared_mem[tid] = -logf(prob + 1e-7f);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        loss[bid] = shared_mem[0] / vocab;
    }
}