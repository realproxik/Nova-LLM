// kernels/part_85_tensor_parallel_allreduce.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tensor_parallel_allreduce_kernel(
    float* input, float* output,
    int size, int rank, int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float sum = 0.0f;
    for (int r = 0; r < world_size; r++) {
        sum += input[r * size + idx];
    }
    output[rank * size + idx] = sum;
}