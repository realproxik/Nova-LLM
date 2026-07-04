// kernels/part_55_gqa.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gqa_kernel(
    float* Q, float* K, float* V, float* O,
    int batch, int heads, int kv_heads, int seq, int dim
) {
    extern __shared__ float shared_mem[];
    int head = blockIdx.z;
    int batch_idx = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int head_dim = dim / heads;
    int kv_head_dim = dim / kv_heads;
    int group_size = heads / kv_heads;
    
    int kv_head = head / group_size;
    
    int base_q = (batch_idx * heads + head) * seq * head_dim;
    int base_kv = (batch_idx * kv_heads + kv_head) * seq * kv_head_dim;
    
    float* Q_base = Q + base_q;
    float* K_base = K + base_kv;
    float* V_base = V + base_kv;
    float* O_base = O + base_q;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + seq * kv_head_dim;
    float* V_block = shared_mem + seq * kv_head_dim * 2;
    
    int row = ty;
    int col = tx;
    
    if (row < seq && col < kv_head_dim) {
        K_block[row * kv_head_dim + col] = K_base[row * kv_head_dim + col];
        V_block[row * kv_head_dim + col] = V_base[row * kv_head_dim + col];
    }
    if (row < seq && col < head_dim) {
        Q_block[row * head_dim + col] = Q_base[row * head_dim + col];
    }
    __syncthreads();
    
    if (row < seq && col < seq) {
        float score = 0.0f;
        for (int i = 0; i < kv_head_dim; i++) {
            score += Q_block[row * head_dim + i] * K_block[col * kv_head_dim + i];
        }
        score /= sqrtf((float)kv_head_dim);
        
        float max_val = score;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < kv_head_dim; j++) {
                temp += Q_block[row * head_dim + j] * K_block[i * kv_head_dim + j];
            }
            temp /= sqrtf((float)kv_head_dim);
            if (temp > max_val) max_val = temp;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < kv_head_dim; j++) {
                temp += Q_block[row * head_dim + j] * K_block[i * kv_head_dim + j];
            }
            temp = expf(temp / sqrtf((float)kv_head_dim) - max_val);
            sum += temp;
        }
        
        for (int i = 0; i < head_dim; i++) {
            float result = 0.0f;
            for (int j = 0; j < seq; j++) {
                float attn = 0.0f;
                for (int k = 0; k < kv_head_dim; k++) {
                    attn += Q_block[row * head_dim + k] * K_block[j * kv_head_dim + k];
                }
                attn = expf(attn / sqrtf((float)kv_head_dim) - max_val) / sum;
                result += attn * V_block[j * kv_head_dim + i];
            }
            O_base[row * head_dim + i] = result;
        }
    }
}