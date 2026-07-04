// kernels/part_81_flash_attn2_forward.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__global__ void flash_attn2_forward_kernel(
    float* Q, float* K, float* V, float* O,
    int batch, int heads, int seq, int dim
) {
    extern __shared__ float shared_mem[];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int block_size = BLOCK_SIZE;
    int head_dim = dim / heads;
    
    int q_start = bx * block_size;
    int kv_start = by * block_size;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + block_size * head_dim;
    float* V_block = shared_mem + block_size * head_dim * 2;
    
    int head = by;
    int batch_idx = bx / ((seq + block_size - 1) / block_size);
    
    int base = (batch_idx * heads + head) * seq * head_dim;
    float* Q_base = Q + base;
    float* K_base = K + base;
    float* V_base = V + base;
    float* O_base = O + base;
    
    if (tx < block_size && ty < head_dim) {
        Q_block[tx * head_dim + ty] = Q_base[(q_start + tx) * head_dim + ty];
        K_block[tx * head_dim + ty] = K_base[(kv_start + tx) * head_dim + ty];
        V_block[tx * head_dim + ty] = V_base[(kv_start + tx) * head_dim + ty];
    }
    __syncthreads();
    
    float max_val = -INFINITY;
    float sum = 0.0f;
    float O_vals[BLOCK_SIZE];
    
    for (int i = 0; i < block_size; i++) {
        O_vals[i] = 0.0f;
    }
    
    for (int k = 0; k < seq; k += block_size) {
        if (kv_start + k < seq) {
            if (tx < block_size && ty < head_dim) {
                K_block[tx * head_dim + ty] = K_base[(k + tx) * head_dim + ty];
                V_block[tx * head_dim + ty] = V_base[(k + tx) * head_dim + ty];
            }
            __syncthreads();
        }
        
        if (tx < block_size && ty < block_size) {
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += Q_block[tx * head_dim + i] * K_block[ty * head_dim + i];
            }
            score /= sqrtf((float)head_dim);
            
            if (score > max_val) max_val = score;
            score = expf(score - max_val);
            sum += score;
            
            for (int i = 0; i < head_dim; i++) {
                O_vals[i] += score * V_block[ty * head_dim + i];
            }
        }
        __syncthreads();
    }
    
    if (tx < block_size && ty < head_dim) {
        O_base[(q_start + tx) * head_dim + ty] = O_vals[ty] / sum;
    }
}