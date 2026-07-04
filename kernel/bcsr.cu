// kernels/part_37_sparse_matmul_block_csr.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void sparse_matmul_block_csr_kernel(
    float* values, int* col_indices, int* row_ptr,
    float* dense, float* output,
    int rows, int cols, int nnz
) {
    __shared__ float dense_block[BLOCK_SIZE][BLOCK_SIZE];
    
    int row_block = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = row_block * BLOCK_SIZE + ty;
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    
    for (int i = start; i < end; i++) {
        int col = col_indices[i];
        float val = values[i];
        
        if (tx == 0) {
            dense_block[ty][0] = dense[col];
        }
        __syncthreads();
        
        sum += val * dense_block[ty][0];
    }
    
    output[row] = sum;
}