// kernels/part_97_context_parallel_allgather.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void context_parallel_allgather_kernel(
    float* input, float* output,
    int size, int context_parallel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    for (int c = 0; c < context_parallel_size; c++) {
        output[c * size + idx] = input[idx];
    }
}