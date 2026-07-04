// kernels/part_26_quant_int4.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void quantize_int4_kernel(float* input, uint8_t* output, float* scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }
    
    float scale_val = 7.0f / max_val;
    *scale = scale_val;
    
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = input[i] * scale_val;
        val = fminf(7.0f, fmaxf(-7.0f, val));
        int8_t quant = (int8_t)roundf(val);
        
        int pack_idx = i / 2;
        int bit_pos = (i % 2) * 4;
        uint8_t packed = output[pack_idx];
        packed &= ~(0x0F << bit_pos);
        packed |= ((uint8_t)(quant & 0x0F) << bit_pos);
        output[pack_idx] = packed;
    }
}