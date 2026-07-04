// kernels/part_99_fused_add_layernorm.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_add_layernorm_kernel(
    float* input1, float* input2, float* output,
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
    
    for (int i = tid; i < dim; i += blockDim.x) {
        output[idx + i] = input1[idx + i] + input2[idx + i];
    }
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += output[idx + i];
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
        float x = output[idx + i] - mean;
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
        output[out_idx] = (output[out_idx] - mean) * inv_std * weight[i] + bias[i];
    }
}