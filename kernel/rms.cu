// kernels/part_13_rms_norm.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(float* input, float* weight, float* output, int batch, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int i = idx / dim;
    int d = idx % dim;
    
    float sum = 0.0f;
    for (int j = 0; j < dim; j++) {
        float val = input[i * dim + j];
        sum += val * val;
    }
    
    float rms = 1.0f / sqrtf(sum / dim + 1e-6f);
    output[idx] = input[idx] * rms * weight[d];
}

__global__ void rms_norm_shared_kernel(float* input, float* weight, float* output, int batch, int seq, int dim) {
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int i = idx / dim;
    
    float sum = 0.0f;
    for (int j = tid; j < dim; j += blockDim.x) {
        float val = input[i * dim + j];
        sum += val * val;
        shared_mem[j] = val;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float total_sum = shared_mem[0];
    float rms = 1.0f / sqrtf(total_sum / dim + 1e-6f);
    
    for (int j = tid; j < dim; j += blockDim.x) {
        output[i * dim + j] = input[i * dim + j] * rms * weight[j];
    }
}