// kernels/part_08_single_head_attention.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void single_head_attention_kernel(float* Q, float* K, float* V, float* O, int seq, int dim) {
    extern __shared__ float shared_mem[];
    float* shared_Q = shared_mem;
    float* shared_K = shared_mem + seq * dim;
    float* shared_V = shared_mem + seq * dim * 2;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    if (row < seq && col < dim) {
        shared_Q[row * dim + col] = Q[row * dim + col];
        shared_K[row * dim + col] = K[row * dim + col];
        shared_V[row * dim + col] = V[row * dim + col];
    }
    __syncthreads();
    
    if (row < seq && col < seq) {
        float score = 0.0f;
        for (int i = 0; i < dim; i++) {
            score += shared_Q[row * dim + i] * shared_K[col * dim + i];
        }
        score /= sqrtf((float)dim);
        
        float max_val = score;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < dim; j++) {
                temp += shared_Q[row * dim + j] * shared_K[i * dim + j];
            }
            temp /= sqrtf((float)dim);
            if (temp > max_val) max_val = temp;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < seq; i++) {
            float temp = 0.0f;
            for (int j = 0; j < dim; j++) {
                temp += shared_Q[row * dim + j] * shared_K[i * dim + j];
            }
            temp = expf(temp / sqrtf((float)dim) - max_val);
            sum += temp;
        }
        
        float result = 0.0f;
        for (int i = 0; i < seq; i++) {
            float attn = 0.0f;
            for (int j = 0; j < dim; j++) {
                attn += shared_Q[row * dim + j] * shared_K[i * dim + j];
            }
            attn = expf(attn / sqrtf((float)dim) - max_val) / sum;
            
            float val = 0.0f;
            for (int j = 0; j < dim; j++) {
                val += attn * shared_V[i * dim + j];
            }
            result = val;
        }
        O[row * dim + col] = result;
    }
}