// kernels/part_75_fp8_quant.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fp8_quant_kernel(
    float* input, uint8_t* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = input[idx];
    
    int sign = val < 0 ? 1 : 0;
    float abs_val = fabsf(val);
    int exponent = 0;
    
    if (abs_val > 0.0f) {
        exponent = (int)floorf(log2f(abs_val)) + 7;
        abs_val = abs_val / powf(2.0f, exponent - 7);
    }
    
    int mantissa = (int)roundf(abs_val * 8.0f);
    mantissa = fminf(7, fmaxf(0, mantissa));
    
    uint8_t fp8 = (sign << 7) | ((exponent & 0x0F) << 3) | (mantissa & 0x07);
    output[idx] = fp8;
}