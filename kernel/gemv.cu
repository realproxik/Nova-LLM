// kernels/part_48_gemv.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gemv_kernel(
    float* A, float* x, float* y,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
        sum += A[row * cols + col] * x[col];
    }
    y[row] = sum;
}

__global__ void gemv_shared_kernel(
    float* A, float* x, float* y,
    int rows, int cols
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tid;
    
    if (row >= rows) return;
    
    float sum = 0.0f;
    for (int block = 0; block < (cols + blockDim.x - 1) / blockDim.x; block++) {
        if (block * blockDim.x + tid < cols) {
            shared_mem[tid] = x[block * blockDim.x + tid];
        }
        __syncthreads();
        
        for (int i = 0; i < blockDim.x; i++) {
            int col = block * blockDim.x + i;
            if (col < cols) {
                sum += A[row * cols + col] * shared_mem[i];
            }
        }
        __syncthreads();
    }
    y[row] = sum;
}