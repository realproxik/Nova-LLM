// kernels/part_96_ulysses_alltoall.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ulysses_alltoall_kernel(
    float* input, float* output,
    int* send_counts, int* send_offsets,
    int* recv_counts, int* recv_offsets,
    int size, int num_ranks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    for (int r = 0; r < num_ranks; r++) {
        for (int i = 0; i < send_counts[r]; i++) {
            int src_idx = send_offsets[r] + i;
            int dst_idx = recv_offsets[r] + i;
            output[dst_idx] = input[src_idx];
        }
    }
}