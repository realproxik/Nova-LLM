// kernels/part_98_ring_allreduce.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ring_allreduce_kernel(
    float* input, float* output,
    int size, int ring_size, int rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = input[rank * size + idx];
    
    for (int i = 0; i < ring_size; i++) {
        int send_rank = (rank + i) % ring_size;
        int recv_rank = (rank - i + ring_size) % ring_size;
        val += input[send_rank * size + idx];
    }
    
    output[rank * size + idx] = val;
}