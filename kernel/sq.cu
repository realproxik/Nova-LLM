// kernels/part_70_smoothquant.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void smoothquant_kernel(
    float* input, float* output,
    float* scale, int* zero_point,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    
    int row = idx / cols;
    int col = idx % cols;
    
    float max_val = 0.0f;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, fabsf(input[row * cols + i]));
    }
    
    float s = 127.0f / max_val;
    scale[row] = s;
    
    float val = input[idx] * s;
    val = fminf(127.0f, fmaxf(-127.0f, val));
    output[idx] = (int8_t)roundf(val);
}