// kernels/part_115_fused_retnet_retention.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_retnet_retention_kernel(
    float* input, float* output,
    float* retention_weights,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = (idx / dim) / seq;
    int s = (idx / dim) % seq;
    int d = idx % seq;
    
    float x = input[idx];
    float rw = retention_weights[(seq - s + d) % seq];
    
    float sum = 0.0f;
    for (int i = 0; i < seq; i++) {
        float w = (i == s) ? 1.0f : retention_weights[(i + d) % seq];
        sum += w * input[b * seq * dim + i * dim + (i + s) % dim];
    }
    
    output[idx] = x + rw * sum;
}