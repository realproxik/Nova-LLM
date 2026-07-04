// kernels/part_43_batch_norm_backward.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void batch_norm_backward_kernel(
    float* grad_output, float* input,
    float* grad_input, float* grad_weight, float* grad_bias,
    float* mean, float* var,
    int batch, int channels, int spatial,
    float eps
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int c = blockIdx.x;
    
    int total = batch * spatial;
    
    float sum_grad = 0.0f;
    float sum_grad_x = 0.0f;
    
    for (int i = tid; i < total; i += blockDim.x) {
        int b = i / spatial;
        int s = i % spatial;
        int idx = b * channels * spatial + c * spatial + s;
        
        float x = input[idx];
        float mu = mean[c];
        float sigma = sqrtf(var[c] + eps);
        float y = (x - mu) / sigma;
        float grad = grad_output[idx];
        
        sum_grad += grad;
        sum_grad_x += grad * y;
    }
    
    shared_mem[tid] = sum_grad;
    shared_mem[tid + blockDim.x] = sum_grad_x;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
            shared_mem[tid + blockDim.x] += shared_mem[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float n = (float)total;
        float sum_g = shared_mem[0];
        float sum_gx = shared_mem[blockDim.x];
        
        float mu = mean[c];
        float sigma = sqrtf(var[c] + eps);
        float gamma = weight[c];
        
        grad_weight[c] = sum_gx / sigma;
        grad_bias[c] = sum_g / n;
    }
}