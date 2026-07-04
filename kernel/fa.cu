// kernels/part_10_flash_attention.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, int seq, int dim) {
    extern __shared__ float shared_mem[];
    
    int q_block = blockIdx.x;
    int kv_block = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int block_size = BLOCK_SIZE;
    int head_dim = dim;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + block_size * head_dim;
    float* V_block = shared_mem + block_size * head_dim * 2;
    float* S_block = shared_mem + block_size * head_dim * 3;
    
    int q_start = q_block * block_size;
    int kv_start = kv_block * block_size;
    
    if (tx < block_size && ty < head_dim) {
        Q_block[tx * head_dim + ty] = Q[(q_start + tx) * head_dim + ty];
        K_block[tx * head_dim + ty] = K[(kv_start + tx) * head_dim + ty];
        V_block[tx * head_dim + ty] = V[(kv_start + tx) * head_dim + ty];
    }
    __syncthreads();
    
    if (tx < block_size && ty < block_size) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += Q_block[tx * head_dim + i] * K_block[ty * head_dim + i];
        }
        S_block[tx * block_size + ty] = score / sqrtf((float)head_dim);
    }
    __syncthreads();
    
    if (tx < block_size && ty < block_size) {
        float max_val = S_block[tx * block_size];
        for (int i = 1; i < block_size; i++) {
            if (S_block[tx * block_size + i] > max_val) {
                max_val = S_block[tx * block_size + i];
            }
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
    
    if (tx < block_size && ty < head_dim) {
        float result = 0.0f;
        for (int i = 0; i < block_size; i++) {
            result += S_block[tx * block_size + i] * V_block[i * head_dim + ty];
        }
        O[(q_start + tx) * head_dim + ty] = result;
    }
}