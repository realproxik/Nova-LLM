// kernels/part_73_ggml_q5_0.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ggml_q5_0_kernel(
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
    
    float scale = max_val / 15.0f;
    output[row * (cols / 2) + col / 2] = (uint8_t)(scale * 100);
    
    int packed_idx = (col * 5) / 8;
    int bit_pos = (col * 5) % 8;
    float val = input[row * cols + col] / scale;
    val = fminf(15.0f, fmaxf(-15.0f, val));
    uint8_t quant = (uint8_t)((int8_t)roundf(val) & 0x1F);
    
    uint8_t packed = output[row * (cols * 5 / 8) + packed_idx];
    uint8_t mask = 0x1F << bit_pos;
    packed &= ~mask;
    packed |= (quant << bit_pos);
    output[row * (cols * 5 / 8) + packed_idx] = packed;
}