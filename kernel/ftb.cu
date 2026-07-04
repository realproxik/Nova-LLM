// kernels/part_121_fused_transformer_block.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_transformer_block_kernel(
    float* input, float* output,
    float* q_weight, float* k_weight, float* v_weight, float* o_weight,
    float* mlp_w1, float* mlp_w2,
    float* norm_weight1, float* norm_weight2,
    int batch, int seq, int hidden, int dim, int intermediate,
    float eps
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid / seq;
    int s = bid % seq;
    int idx = b * seq * hidden + s * hidden;
    
    // RMSNorm 1
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
    
    float rms1 = 1.0f / sqrtf(shared_mem[0] / hidden + eps);
    
    float norm_input1[4096];
    for (int i = tid; i < hidden; i += blockDim.x) {
        norm_input1[i] = input[idx + i] * rms1 * norm_weight1[i];
    }
    __syncthreads();
    
    // Attention
    float attn_out[4096];
    for (int h = 0; h < dim; h++) {
        float q = 0.0f, k = 0.0f, v = 0.0f;
        for (int i = 0; i < hidden; i++) {
            q += norm_input1[i] * q_weight[i * dim + h];
            k += norm_input1[i] * k_weight[i * dim + h];
            v += norm_input1[i] * v_weight[i * dim + h];
        }
        float score = q * k / sqrtf((float)dim);
        float attn = expf(score) / (1.0f + expf(score));
        attn_out[h] = attn * v + input[idx + h];
    }
    __syncthreads();
    
    // RMSNorm 2
    sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float x = attn_out[i];
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
    
    float rms2 = 1.0f / sqrtf(shared_mem[0] / hidden + eps);
    
    float norm_input2[4096];
    for (int i = tid; i < hidden; i += blockDim.x) {
        norm_input2[i] = attn_out[i] * rms2 * norm_weight2[i];
    }
    __syncthreads();
    
    // MLP
    float temp[4096 * 4];
    for (int i = 0; i < intermediate; i++) {
        float sum_val = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum_val += norm_input2[j] * mlp_w1[i * hidden + j];
        }
        temp[i] = sum_val > 0 ? sum_val : 0;
    }
    __syncthreads();
    
    float mlp_out[4096];
    for (int h = 0; h < hidden; h++) {
        float sum_val = 0.0f;
        for (int i = 0; i < intermediate; i++) {
            sum_val += temp[i] * mlp_w2[h * intermediate + i];
        }
        mlp_out[h] = sum_val;
    }
    __syncthreads();
    
    // Residual
    for (int i = tid; i < hidden; i += blockDim.x) {
        output[idx + i] = attn_out[i] + mlp_out[i];
    }
}