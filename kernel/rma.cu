// kernels/part_102_fused_rmsnorm_attn.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_rmsnorm_attn_kernel(
    float* input, float* output,
    float* norm_weight,
    float* q_weight, float* k_weight, float* v_weight,
    int batch, int seq, int hidden, int dim,
    float eps
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid / seq;
    int s = bid % seq;
    int idx = b * seq * hidden + s * hidden;
    
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
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
    
    float rms = 1.0f / sqrtf(shared_mem[0] / hidden + eps);
    
    float norm_input[4096];
    for (int i = tid; i < hidden; i += blockDim.x) {
        norm_input[i] = input[idx + i] * rms * norm_weight[i];
    }
    __syncthreads();
    
    // Attention with normalized input
    for (int h = 0; h < dim; h++) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        for (int i = 0; i < hidden; i++) {
            q_val += norm_input[i] * q_weight[i * dim + h];
            k_val += norm_input[i] * k_weight[i * dim + h];
            v_val += norm_input[i] * v_weight[i * dim + h];
        }
        output[idx + h] = q_val + k_val + v_val;
    }
}