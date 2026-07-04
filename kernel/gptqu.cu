// kernels/part_68_dequant_gptq.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dequant_gptq_kernel(
    int8_t* input, float* output,
    float* scale, int* zero_point,
    int rows, int cols, int group_size
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    int group = col / group_size;
    float s = scale[row * (cols / group_size) + group];
    int zp = zero_point[row * (cols / group_size) + group];
    
    output[row * cols + col] = ((float)input[row * cols + col] - zp) * s;
}