// kernels/part_110_fused_flash_rope.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void fused_flash_rope_kernel(
    float* Q, float* K, float* V, float* O,
    int batch, int heads, int seq, int dim,
    float theta
) {
    extern __shared__ float shared_mem[];
    int q_block = blockIdx.x;
    int kv_block = blockIdx.y;
    int head = blockIdx.z;
    int batch_idx = blockIdx.w;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int block_size = BLOCK_SIZE;
    int head_dim = dim / heads;
    int base = (batch_idx * heads + head) * seq * head_dim;
    
    float* Q_base = Q + base;
    float* K_base = K + base;
    float* V_base = V + base;
    float* O_base = O + base;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + block_size * head_dim;
    float* V_block = shared_mem + block_size * head_dim * 2;
    float* S_block = shared_mem + block_size * head_dim * 3;
    
    int q_start = q_block * block_size;
    int kv_start = kv_block * block_size;
    
    if (tx < block_size && ty < head_dim) {
        Q_block[tx * head_dim + ty] = Q_base[(q_start + tx) * head_dim + ty];
        K_block[tx * head_dim + ty] = K_base[(kv_start + tx) * head_dim + ty];
        V_block[tx * head_dim + ty] = V_base[(kv_start + tx) * head_dim + ty];
    }
    __syncthreads();
    
    // Apply RoPE
    if (tx < block_size && ty < head_dim / 2) {
        float angle = (float)(q_start + tx) / powf(theta, (float)ty / head_dim);
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float q_val = Q_block[tx * head_dim + ty];
        float q_shift = Q_block[tx * head_dim + ty + head_dim / 2];
        Q_block[tx * head_dim + ty] = q_val * cos_val - q_shift * sin_val;
        Q_block[tx * head_dim + ty + head_dim / 2] = q_val * sin_val + q_shift * cos_val;
        
        float angle2 = (float)(kv_start + tx) / powf(theta, (float)ty / head_dim);
        float cos_val2 = cosf(angle2);
        float sin_val2 = sinf(angle2);
        
        float k_val = K_block[tx * head_dim + ty];
        float k_shift = K_block[tx * head_dim + ty + head_dim / 2];
        K_block[tx * head_dim + ty] = k_val * cos_val2 - k_shift * sin_val2;
        K_block[tx * head_dim + ty + head_dim / 2] = k_val * sin_val2 + k_shift * cos_val2;
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
    
    if (tx < block_size && ty < head_dim) {
        float result = 0.0f;
        for (int i = 0; i < block_size; i++) {
            result += S_block[tx * block_size + i] * V_block[i * head_dim + ty];
        }
        O_base[(q_start + tx) * head_dim + ty] = result;
    }
}