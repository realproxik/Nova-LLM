// kernels/part_74_ggml_q8_0.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ggml_q8_0_kernel(
    float* input, int8_t* output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    float max_val = 0.0f;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, fabsf(input[row * cols + i]));
    }
    
    float scale = max_val / 127.0f;
    output[row * cols + col] = (int8_t)roundf(input[row * cols + col] / scale);
}