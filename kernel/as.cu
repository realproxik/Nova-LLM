// kernels/part_63_cross_entropy_fp16.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void cross_entropy_fp16_kernel(
    half* logits, int* targets, float* loss,
    int batch, int vocab
) {
    extern __shared__ half shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid;
    
    half max_val = logits[b * vocab];
    for (int i = 1; i < vocab; i++) {
        max_val = __hmax(max_val, logits[b * vocab + i]);
    }
    
    half sum = __float2half(0.0f);
    for (int i = 0; i < vocab; i++) {
        half val = __hexp(__hsub(logits[b * vocab + i], max_val));
        sum = __hadd(sum, val);
    }
    
    int target = targets[b];
    half prob = __hdiv(__hexp(__hsub(logits[b * vocab + target], max_val)), sum);
    
    shared_mem[tid] = __float2half(-logf(__half2float(prob) + 1e-7f));
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = __hadd(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        loss[bid] = __half2float(shared_mem[0]) / vocab;
    }
}