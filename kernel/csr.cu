// kernels/part_35_sparse_matmul_csr.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sparse_matmul_csr_kernel(
    float* values, int* col_indices, int* row_ptr,
    float* dense, float* output,
    int rows, int cols, int nnz
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    for (int i = start; i < end; i++) {
        int col = col_indices[i];
        sum += values[i] * dense[col];
    }
    
    output[row] = sum;
}