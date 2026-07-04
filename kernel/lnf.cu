// kernels/part_44_layer_norm_forward.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void layer_norm_forward_kernel(
    float* input, float* output,
    float* weight, float* bias,
    int batch, int seq, int dim,
    float eps
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid / seq;
    int s = bid % seq;
    int idx = b * seq * dim + s * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += input[idx + i];
    }
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / dim;
    
    float var_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float x = input[idx + i] - mean;
        var_sum += x * x;
    }
    shared_mem[tid] = var_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float var = shared_mem[0] / dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        int out_idx = idx + i;
        output[out_idx] = (input[out_idx] - mean) * inv_std * weight[i] + bias[i];
    }
}