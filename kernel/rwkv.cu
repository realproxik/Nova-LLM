// kernels/part_114_fused_rwkv_attn.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_rwkv_attn_kernel(
    float* input, float* output,
    float* time_decay, float* time_first,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = (idx / dim) / seq;
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    float x = input[idx];
    float td = time_decay[d];
    float tf = time_first[d];
    
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    for (int i = 0; i <= s; i++) {
        float w = (i == s) ? tf : expf(-td * (s - i));
        numerator += w * input[b * seq * dim + i * dim + d];
        denominator += w;
    }
    
    output[idx] = numerator / (denominator + 1e-7f);
}