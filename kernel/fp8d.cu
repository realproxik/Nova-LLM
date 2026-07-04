// kernels/part_76_fp8_dequant.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fp8_dequant_kernel(
    uint8_t* input, float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    uint8_t fp8 = input[idx];
    int sign = (fp8 >> 7) & 0x01;
    int exponent = (fp8 >> 3) & 0x0F;
    int mantissa = fp8 & 0x07;
    
    float val = (1.0f + mantissa / 8.0f) * powf(2.0f, exponent - 7);
    if (sign) val = -val;
    
    output[idx] = val;
}