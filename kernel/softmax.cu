// kernels/part_18_softmax.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* input, float* output, int batch, int seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch * seq) return;
    
    int b = idx / seq;
    int s = idx % seq;
    
    float max_val = input[idx];
    for (int i = 0; i < seq; i++) {
        if (input[b * seq + i] > max_val) max_val = input[b * seq + i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < seq; i++) {
        sum += expf(input[b * seq + i] - max_val);
    }
    
    output[idx] = expf(input[idx] - max_val) / sum;
}

__global__ void softmax_shared_kernel(float* input, float* output, int batch, int seq) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid;
    int s = tid;
    
    if (s < seq) {
        shared_mem[tid] = input[b * seq + s];
    }
    __syncthreads();
    
    float max_val = shared_mem[0];
    for (int i = 1; i < seq; i++) {
        if (shared_mem[i] > max_val) max_val = shared_mem[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < seq; i++) {
        if (i == s) {
            shared_mem[i] = expf(shared_mem[i] - max_val);
        }
        sum += shared_mem[i];
    }
    __syncthreads();
    
    if (s < seq) {
        output[b * seq + s] = shared_mem[s] / sum;
    }
}