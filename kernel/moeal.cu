// kernels/part_61_moe_aux_loss.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_aux_loss_kernel(
    int* expert_indices,
    float* expert_scores,
    float* aux_loss,
    int batch, int seq, int num_experts, int top_k
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int e = blockIdx.x;
    
    float count = 0.0f;
    float sum = 0.0f;
    
    int total = batch * seq;
    for (int i = tid; i < total; i += blockDim.x) {
        for (int k = 0; k < top_k; k++) {
            if (expert_indices[i * top_k + k] == e) {
                count += 1.0f;
                sum += expert_scores[i * top_k + k];
            }
        }
    }
    
    shared_mem[tid] = count;
    shared_mem[tid + blockDim.x] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
            shared_mem[tid + blockDim.x] += shared_mem[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float total_count = shared_mem[0];
        float total_sum = shared_mem[blockDim.x];
        aux_loss[e] = total_sum * total_count / (total * top_k);
    }
}