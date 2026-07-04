// kernels/part_78_kv_cache_copy.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kv_cache_copy_kernel(
    float* src_k, float* src_v,
    float* dst_k, float* dst_v,
    int batch, int heads, int seq, int dim,
    int src_offset, int dst_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * dim;
    
    if (idx >= total) return;
    
    int b = idx / (heads * seq * dim);
    int h = (idx / (seq * dim)) % heads;
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    int src_idx = b * heads * seq * dim + h * seq * dim + (s + src_offset) * dim + d;
    int dst_idx = b * heads * seq * dim + h * seq * dim + (s + dst_offset) * dim + d;
    
    dst_k[dst_idx] = src_k[src_idx];
    dst_v[dst_idx] = src_v[src_idx];
}