// kernels/part_42_batch_norm_forward.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void batch_norm_forward_kernel(
    float* input, float* output,
    float* mean, float* var,
    float* weight, float* bias,
    int batch, int channels, int spatial,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * spatial;
    
    if (idx >= total) return;
    
    int b = idx / (channels * spatial);
    int c = (idx / spatial) % channels;
    int s = idx % spatial;
    
    float x = input[idx];
    float mu = mean[c];
    float sigma = sqrtf(var[c] + eps);
    float gamma = weight[c];
    float beta = bias[c];
    
    output[idx] = (x - mu) / sigma * gamma + beta;
}