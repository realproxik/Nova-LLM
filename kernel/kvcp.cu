// kernels/part_84_kv_cache_prefetch.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kv_cache_prefetch_kernel(
    float* kv_cache, int* token_indices,
    int* prefetch_indices, int num_prefetch,
    int batch, int heads, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_prefetch) return;
    
    int token = token_indices[idx];
    int prefetch = prefetch_indices[idx];
    
    if (token < seq) {
        __builtin_prefetch(&kv_cache[token * batch * heads * dim], 0, 3);
        __builtin_prefetch(&kv_cache[prefetch * batch * heads * dim], 0, 3);
    }
}