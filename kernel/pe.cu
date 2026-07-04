// kernels/part_22_positional_embedding.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void positional_embedding_kernel(float* input, float* output, int batch, int seq, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    
    if (idx >= total) return;
    
    int b = idx / (seq * dim);
    int s = (idx / dim) % seq;
    int d = idx % dim;
    
    int output_idx = b * seq * dim + s * dim + d;
    
    if (d % 2 == 0) {
        float angle = (float)s / powf(10000.0f, (float)d / dim);
        output[output_idx] = input[output_idx] + sinf(angle);
    } else {
        float angle = (float)s / powf(10000.0f, (float)(d - 1) / dim);
        output[output_idx] = input[output_idx] + cosf(angle);
    }
}