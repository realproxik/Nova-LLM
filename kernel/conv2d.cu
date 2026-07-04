// kernels/part_39_conv2d.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    float* input, float* kernel, float* output,
    int batch, int in_channels, int out_channels,
    int height, int width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int b = blockIdx.z;
    int out_c = blockIdx.y;
    int out_h = blockIdx.x / ((width + blockDim.x - 1) / blockDim.x);
    int out_w = (blockIdx.x % ((width + blockDim.x - 1) / blockDim.x)) * blockDim.x + threadIdx.x;
    
    if (out_h >= height || out_w >= width) return;
    
    float sum = 0.0f;
    int in_h_start = out_h * stride_h - pad_h;
    int in_w_start = out_w * stride_w - pad_w;
    
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int in_h = in_h_start + kh;
                int in_w = in_w_start + kw;
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    sum += input[b * in_channels * height * width + in_c * height * width + in_h * width + in_w] *
                           kernel[out_c * in_channels * kernel_h * kernel_w + in_c * kernel_h * kernel_w + kh * kernel_w + kw];
                }
            }
        }
    }
    
    output[b * out_channels * height * width + out_c * height * width + out_h * width + out_w] = sum;
}