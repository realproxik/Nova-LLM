// kernels/part_14_layer_norm.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(float* input, float* weight, float* bias, float* output, int batch, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int i = idx / dim;
    int d = idx % dim;
    
    float mean = 0.0f;
    for (int j = 0; j < dim; j++) {
        mean += input[i * dim + j];
    }
    mean /= dim;
    
    float var = 0.0f;
    for (int j = 0; j < dim; j++) {
        float val = input[i * dim + j] - mean;
        var += val * val;
    }
    var /= dim;
    
    float std = sqrtf(var + 1e-6f);
    output[idx] = (input[idx] - mean) / std * weight[d] + bias[d];
}

__global__ void layer_norm_fused_kernel(float* input, float* weight, float* bias, float* output, int batch, int seq, int dim) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x;
    
    int i = idx;
    
    float sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float val = input[i * dim + j];
        sum += val;
        shared_mem[j] = val;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_mem[0] / dim;
    
    float var_sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float val = shared_mem[j] - mean;
        var_sum += val * val;
        shared_mem[j] = val;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float std = sqrtf(shared_mem[0] / dim + 1e-6f);
    
    for (int j = tid; j < dim; j += blockDim.x) {
        output[i * dim + j] = shared_mem[j] / std * weight[j] + bias[j];
    }
}