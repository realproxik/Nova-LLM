// kernels/part_72_ggml_q4_1.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ggml_q4_1_kernel(
    float* input, uint8_t* output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    float min_val = INFINITY;
    float max_val = -INFINITY;
    for (int i = 0; i < cols; i++) {
        float val = input[row * cols + i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    float scale = (max_val - min_val) / 15.0f;
    output[row * (cols / 2) + col / 2] = (uint8_t)(scale * 100);
    output[row * (cols / 2) + col / 2 + (rows * cols / 2)] = (uint8_t)(min_val * 100);
    
    int packed_idx = col / 2;
    int bit_pos = (col % 2) * 4;
    float val = (input[row * cols + col] - min_val) / scale;
    val = fminf(15.0f, fmaxf(0.0f, val));
    uint8_t quant = (uint8_t)roundf(val);
    uint8_t packed = output[row * (cols / 2) + packed_idx];
    packed &= ~(0x0F << bit_pos);
    packed |= (quant << bit_pos);
    output[row * (cols / 2) + packed_idx] = packed;
}