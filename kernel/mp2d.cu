// kernels/part_40_maxpool2d.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(
    float* input, float* output,
    int batch, int channels,
    int height, int width,
    int pool_h, int pool_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int out_h = blockIdx.x / ((width + blockDim.x - 1) / blockDim.x);
    int out_w = (blockIdx.x % ((width + blockDim.x - 1) / blockDim.x)) * blockDim.x + threadIdx.x;
    
    if (out_h >= height || out_w >= width) return;
    
    int in_h_start = out_h * stride_h - pad_h;
    int in_w_start = out_w * stride_w - pad_w;
    
    float max_val = -INFINITY;
    for (int kh = 0; kh < pool_h; kh++) {
        for (int kw = 0; kw < pool_w; kw++) {
            int in_h = in_h_start + kh;
            int in_w = in_w_start + kw;
            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                float val = input[b * channels * height * width + c * height * width + in_h * width + in_w];
                if (val > max_val) max_val = val;
            }
        }
    }
    
    output[b * channels * height * width + c * height * width + out_h * width + out_w] = max_val;
}