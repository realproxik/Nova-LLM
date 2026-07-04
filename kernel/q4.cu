// kernels/part_71_ggml_q4_0.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ggml_q4_0_kernel(
    float* input, uint8_t* output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    float max_val = 0.0f;
    for (int i = 0; i < cols; i++) {
        max_val = fmaxf(max_val, fabsf(input[row * cols + i]));
    }
    
    float scale = max_val / 7.0f;
    output[row * (cols / 2) + col / 2] = (uint8_t)(scale * 100);
    
    int packed_idx = col / 2;
    int bit_pos = (col % 2) * 4;
    float val = input[row * cols + col] / scale;
    val = fminf(7.0f, fmaxf(-7.0f, val));
    uint8_t quant = (uint8_t)((int8_t)roundf(val) & 0x0F);
    uint8_t packed = output[row * (cols / 2) + packed_idx];
    packed &= ~(0x0F << bit_pos);
    packed |= (quant << bit_pos);
    output[row * (cols / 2) + packed_idx] = packed;
}