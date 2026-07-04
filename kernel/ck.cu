// kernels/part_107_fused_grad_checkpoint.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_grad_checkpoint_kernel(
    float* input, float* grad_input,
    float* checkpoint, float* grad_checkpoint,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    checkpoint[idx] = input[idx];
    grad_input[idx] = grad_checkpoint[idx];
}