// kernels/part_69_awq_quant.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void awq_quant_kernel(
    float* weights, float* scales,
    int8_t* quantized_weights,
    int rows, int cols, int group_size
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    int group = col / group_size;
    float max_val = 0.0f;
    
    for (int i = 0; i < group_size && group * group_size + i < cols; i++) {
        max_val = fmaxf(max_val, fabsf(weights[row * cols + group * group_size + i]));
    }
    
    float scale = 127.0f / max_val;
    scales[row * (cols / group_size) + group] = scale;
    
    float val = weights[row * cols + col] * scale;
    val = fminf(127.0f, fmaxf(-127.0f, val));
    quantized_weights[row * cols + col] = (int8_t)roundf(val);
}