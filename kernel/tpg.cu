// kernels/part_86_tensor_parallel_allgather.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tensor_parallel_allgather_kernel(
    float* input, float* output,
    int size, int rank, int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    for (int r = 0; r < world_size; r++) {
        output[r * size + idx] = input[r * size + idx];
    }
}