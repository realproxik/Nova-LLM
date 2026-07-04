// kernels/part_05_matmul_bf16.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TILE_SIZE 16

__global__ void matmul_bf16_kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    __shared__ __nv_bfloat16 As[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < M && k * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        }
        
        if (col < N && k * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += __bfloat162float(As[ty][i]) * __bfloat162float(Bs[i][tx]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}

extern "C" void matmul_bf16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
    __nv_bfloat16 *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));
    
    cudaMemcpy(d_A, A, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_bf16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(C, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}