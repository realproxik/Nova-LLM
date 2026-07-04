// kernels/part_80_continuous_batching.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void continuous_batching_kernel(
    float* input, float* output,
    int* sequence_lengths, int* batch_indices,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = idx / (seq * dim);
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    int orig_b = batch_indices[b];
    
    if (s < sequence_lengths[b]) {
        output[orig_b * seq * dim + s * dim + d] = input[b * seq * dim + s * dim + d];
    }
}