// kernels/part_92_context_parallel_allreduce.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void context_parallel_allreduce_kernel(
    float* input, float* output,
    int size, int context_parallel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float sum = 0.0f;
    for (int c = 0; c < context_parallel_size; c++) {
        sum += input[c * size + idx];
    }
    output[idx] = sum;
}