// kernels/part_04_matmul_fp16.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void matmul_fp16_kernel(half* A, half* B, half* C, int M, int N, int K) {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    half sum = __float2half(0.0f);
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < M && k * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        }
        
        if (col < N && k * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum = __hadd(sum, __hmul(As[ty][i], Bs[i][tx]));
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void matmul_fp16(half* A, half* B, half* C, int M, int N, int K) {
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}