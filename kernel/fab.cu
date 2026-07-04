// kernels/part_47_flash_attention_block.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void flash_attention_block_kernel(
    float* Q, float* K, float* V, float* O,
    int seq, int dim,
    int q_start, int kv_start, int block_size
) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + block_size * dim;
    float* V_block = shared_mem + block_size * dim * 2;
    float* S_block = shared_mem + block_size * dim * 3;
    
    if (tx < block_size && ty < dim) {
        Q_block[tx * dim + ty] = Q[(q_start + tx) * dim + ty];
        K_block[tx * dim + ty] = K[(kv_start + tx) * dim + ty];
        V_block[tx * dim + ty] = V[(kv_start + tx) * dim + ty];
    }
    __syncthreads();
    
    if (tx < block_size && ty < block_size) {
        float score = 0.0f;
        for (int i = 0; i < dim; i++) {
            score += Q_block[tx * dim + i] * K_block[ty * dim + i];
        }
        S_block[tx * block_size + ty] = score / sqrtf((float)dim);
    }
    __syncthreads();
    
    if (tx < block_size && ty < block_size) {
        float max_val = S_block[tx * block_size];
        for (int i = 1; i < block_size; i++) {
            if (S_block[tx * block_size + i] > max_val) max_val = S_block[tx * block_size + i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < block_size; i++) {
            S_block[tx * block_size + i] = expf(S_block[tx * block_size + i] - max_val);
            sum += S_block[tx * block_size + i];
        }
        
        for (int i = 0; i < block_size; i++) {
            S_block[tx * block_size + i] /= sum;
        }
    }
    __syncthreads();
    
    if (tx < block_size && ty < dim) {
        float result = 0.0f;
        for (int i = 0; i < block_size; i++) {
            result += S_block[tx * block_size + i] * V_block[i * dim + ty];
        }
        O[(q_start + tx) * dim + ty] = result;
    }
}