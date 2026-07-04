// kernels/part_113_fused_mamba_ssm.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_mamba_ssm_kernel(
    float* input, float* output,
    float* A, float* B, float* C,
    float* delta,
    int batch, int seq, int dim,
    int state_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = (idx / dim) / seq;
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    float x = input[idx];
    float dt = delta[idx];
    
    float state[16] = {0.0f};
    float a = A[d];
    float b = B[d];
    float c = C[d];
    
    state[d % state_dim] = state[d % state_dim] * expf(dt * a) + b * x;
    output[idx] = c * state[d % state_dim];
}