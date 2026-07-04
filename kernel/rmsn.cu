// kernels/part_62_rms_norm_fp16.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void rms_norm_fp16_kernel(
    half* input, half* weight, half* output,
    int batch, int seq, int dim, float eps
) {
    extern __shared__ half shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid / seq;
    int s = bid % seq;
    int idx = b * seq * dim + s * dim;
    
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(input[idx + i]);
        sum += val * val;
    }
    shared_mem[tid] = __float2half(sum);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = __hadd(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    float rms = 1.0f / sqrtf(__half2float(shared_mem[0]) / dim + eps);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        int out_idx = idx + i;
        float val = __half2float(input[out_idx]);
        float w = __half2float(weight[i]);
        output[out_idx] = __float2half(val * rms * w);
    }
}