// kernels/part_56_mqa.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mqa_kernel(
    float* Q, float* K, float* V, float* O,
    int batch, int heads, int seq, int dim
) {
    extern __shared__ float shared_mem[];
    int head = blockIdx.z;
    int batch_idx = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int head_dim = dim / heads;
    int base = (batch_idx * heads + head) * seq * head_dim;
    int kv_base = batch_idx * seq * head_dim;
    
    float* Q_base = Q + base;
    float* K_base = K + kv_base;
    float* V_base = V + kv_base;
    float* O_base = O + base;
    
    float* Q_block = shared_mem;
    float* K_block = shared_mem + seq * head_dim;
    float* V_block = shared_mem + seq * head_dim * 2;
    
    int row = ty;
    int col = tx;
    
    if (row < seq && col < head_dim) {
        Q_block[row * head_dim + col] = Q_base[row * head_dim + col];
        K_block[row * head_dim + col] = K_base[row * head_dim + col];
        V_block[row * head_dim + col] = V_base[row * head_dim + col];
    }
    __syncthreads();
    
    if (row < seq && col < seq) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += Q_block[row * head_dim + i] * K_block[col * head_dim + i];
        }
        score /= sqrtf((float)head_dim);
        
        float max_val = score;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                temp += Q_block[row * head_dim + j] * K_block[i * head_dim + j];
            }
            temp /= sqrtf((float)head_dim);
            if (temp > max_val) max_val = temp;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                temp += Q_block[row * head_dim + j] * K_block[i * head_dim + j];
            }
            temp = expf(temp / sqrtf((float)head_dim) - max_val);
            sum += temp;
        }
        
        for (int i = 0; i < head_dim; i++) {
            float result = 0.0f;
            for (int j = 0; j < seq; j++) {
                float attn = 0.0f;
                for (int k = 0; k < head_dim; k++) {
                    attn += Q_block[row * head_dim + k] * K_block[j * head_dim + k];
                }
                attn = expf(attn / sqrtf((float)head_dim) - max_val) / sum;
                result += attn * V_block[j * head_dim + i];
            }
            O_base[row * head_dim + i] = result;
        }
    }
}