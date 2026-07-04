// kernels/part_06_matmul_int8.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_int8_kernel(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int sum = 0;
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < M && k * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        }
        
        if (col < N && k * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += (int)As[ty][i] * (int)Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void matmul_int8(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));
    
    cudaMemcpy(d_A, A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_int8_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}