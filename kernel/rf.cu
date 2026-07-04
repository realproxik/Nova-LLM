// kernels/part_46_rope_fused.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rope_fused_kernel(
    float* q, float* k,
    int batch, int heads, int seq, int dim,
    float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * dim;
    
    if (idx >= total) return;
    
    int pos = (idx / dim) % seq;
    int d = idx % dim;
    int head_dim = dim / 2;
    
    if (d < head_dim) {
        float angle = (float)pos / powf(theta, (float)d / head_dim);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float q_val = q[idx];
        float q_shift = q[idx + head_dim];
        q[idx] = q_val * cos_val - q_shift * sin_val;
        q[idx + head_dim] = q_val * sin_val + q_shift * cos_val;
        
        float k_val = k[idx];
        float k_shift = k[idx + head_dim];
        k[idx] = k_val * cos_val - k_shift * sin_val;
        k[idx + head_dim] = k_val * sin_val + k_shift * cos_val;
    }
}