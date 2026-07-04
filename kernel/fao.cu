// kernels/part_11_flash_attention_online.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void flash_attention_online_kernel(float* Q, float* K, float* V, float* O, int seq, int dim) {
    extern __shared__ float shared_mem[];
    
    int q_block = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int block_size = BLOCK_SIZE;
    int head_dim = dim;
    int num_kv_blocks = (seq + block_size - 1) / block_size;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + block_size * head_dim;
    float* V_block = shared_mem + block_size * head_dim * 2;
    float* S_block = shared_mem + block_size * head_dim * 3;
    
    int q_start = q_block * block_size;
    
    if (tx < block_size && ty < head_dim) {
        Q_block[tx * head_dim + ty] = Q[(q_start + tx) * head_dim + ty];
    }
    __syncthreads();
    
    float max_val = -INFINITY;
    float sum = 0.0f;
    float O_block[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * block_size;
        
        if (tx < block_size && ty < head_dim) {
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
        
        if (tx < block_size) {
            float row_max = -INFINITY;
            for (int i = 0; i < block_size; i++) {
                if (S_block[tx * block_size + i] > row_max) {
                    row_max = S_block[tx * block_size + i];
                }
            }
            
            float row_sum = 0.0f;
            for (int i = 0; i < block_size; i++) {
                S_block[tx * block_size + i] = expf(S_block[tx * block_size + i] - row_max);
                row_sum += S_block[tx * block_size + i];
            }
            
            float new_max = max_val > row_max ? max_val : row_max;
            float exp_diff = expf(max_val - new_max);
            float exp_diff2 = expf(row_max - new_max);
            
            for (int i = 0; i < block_size; i++) {
                if (ty < head_dim) {
                    O_block[tx][ty] = O_block[tx][ty] * exp_diff + 
                                      S_block[tx * block_size + i] * V_block[i * head_dim + ty] * exp_diff2;
                }
            }
            
            sum = sum * exp_diff + row_sum * exp_diff2;
            max_val = new_max;
        }
        __syncthreads();
    }
    
    if (tx < block_size && ty < head_dim) {
        O[(q_start + tx) * head_dim + ty] = O_block[tx][ty] / sum;
    }
}