// kernels/part_108_fused_prefix_lm.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_prefix_lm_kernel(
    float* input, float* output,
    int* prefix_mask,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = idx / (seq * dim);
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    if (prefix_mask[b * seq + s]) {
        output[idx] = input[idx];
    }
}