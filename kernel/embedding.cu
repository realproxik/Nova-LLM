// kernels/part_20_dropout.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dropout_kernel(float* input, float* output, float* mask, float rate, int size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    unsigned int state = seed + idx * 1103515245 + 12345;
    float random = (float)(state & 0x7fffffff) / (float)0x7fffffff;
    
    float keep = 1.0f - rate;
    if (random < keep) {
        output[idx] = input[idx] / keep;
        mask[idx] = 1.0f;
    } else {
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    }
}