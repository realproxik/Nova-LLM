// kernels/part_50_outer_product.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void outer_product_kernel(
    float* a, float* b, float* c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= size || idy >= size) return;
    
    c[idx * size + idy] = a[idx] * b[idy];
}