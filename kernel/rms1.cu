// kernels/part_45_rms_norm_fused.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rms_norm_fused_kernel(
    float* input, float* output,
    float* weight,
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
        float x = input[idx + i];
        sum += x * x;
    }
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = 1.0f / sqrtf(shared_mem[0] / dim + eps);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        int out_idx = idx + i;
        output[out_idx] = input[out_idx] * rms * weight[i];
    }
}