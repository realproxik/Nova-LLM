// kernels/part_07_matmul_int4.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define INT4_PACK 2

__device__ inline int8_t unpack_int4(uint8_t packed, int idx) {
    return (idx == 0) ? (int8_t)(packed & 0x0F) : (int8_t)((packed >> 4) & 0x0F);
}

__global__ void matmul_int4_kernel(uint8_t* A, uint8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int sum = 0;
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        if (row < M && k * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * (K / INT4_PACK) + (k * TILE_SIZE + tx) / INT4_PACK];
        }
        
        if (col < N && k * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[((k * TILE_SIZE + ty) / INT4_PACK) * N + col];
        }
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; i++) {
            int a_idx = k * TILE_SIZE + i;
            int b_idx = k * TILE_SIZE + i;
            int8_t a_val = unpack_int4(As[ty][i], a_idx % INT4_PACK);
            int8_t b_val = unpack_int4(Bs[i][tx], b_idx % INT4_PACK);
            sum += (int)a_val * (int)b_val;
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}