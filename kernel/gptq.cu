// kernels/part_67_quant_gptq.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void quant_gptq_kernel(
    float* input, int8_t* output,
    float* scale, int* zero_point,
    int rows, int cols, int group_size
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    int group = col / group_size;
    float s = scale[row * (cols / group_size) + group];
    int zp = zero_point[row * (cols / group_size) + group];
    
    float val = input[row * cols + col] / s + zp;
    val = fminf(127.0f, fmaxf(-127.0f, val));
    output[row * cols + col] = (int8_t)roundf(val);
}