// kernels/part_38_conv1d.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    float* input, float* kernel, float* output,
    int batch, int in_channels, int out_channels,
    int width, int kernel_size, int stride, int padding
) {
    int b = blockIdx.z;
    int out_c = blockIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_x >= width) return;
    
    float sum = 0.0f;
    int in_x_start = out_x * stride - padding;
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_x = in_x_start + k;
            if (in_x >= 0 && in_x < width) {
                sum += input[b * in_channels * width + in_c * width + in_x] *
                       kernel[out_c * in_channels * kernel_size + in_c * kernel_size + k];
            }
        }
    }
    
    output[b * out_channels * width + out_c * width + out_x] = sum;
}