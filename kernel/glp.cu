// kernels/part_116_fused_glm_prefix.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_glm_prefix_kernel(
    float* input, float* output,
    int* prefix_mask, int* prefix_length,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = idx / (seq * dim);
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    int prefix_len = prefix_length[b];
    int is_prefix = (s < prefix_len);
    
    if (is_prefix) {
        output[idx] = input[idx];
    } else {
        float sum = 0.0f;
        for (int i = 0; i < prefix_len; i++) {
            sum += input[b * seq * dim + i * dim + d];
        }
        output[idx] = input[idx] + sum / prefix_len;
    }
}