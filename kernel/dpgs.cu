// kernels/part_88_data_parallel_grad_sync.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void data_parallel_grad_sync_kernel(
    float* grad, float* global_grad,
    int size, int rank, int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float sum = 0.0f;
    for (int r = 0; r < world_size; r++) {
        sum += grad[r * size + idx];
    }
    global_grad[rank * size + idx] = sum / world_size;
}