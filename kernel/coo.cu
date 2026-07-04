// kernels/part_36_sparse_matmul_coo.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sparse_matmul_coo_kernel(
    float* values, int* row_indices, int* col_indices,
    float* dense, float* output,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    
    int row = row_indices[idx];
    int col = col_indices[idx];
    float val = values[idx];
    
    atomicAdd(&output[row], val * dense[col]);
}