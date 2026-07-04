// kernels/part_51_batched_matmul.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batched_matmul_kernel(
    float* A, float* B, float* C,
    int batch, int M, int N, int K
) {
    int b = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float* A_b = A + b * M * K;
    float* B_b = B + b * K * N;
    float* C_b = C + b * M * N;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < M && k * TILE_SIZE + tx < K) {
            As[ty][tx] = A_b[row * K + k * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && k * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B_b[(k * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C_b[row * N + col] = sum;
    }
}