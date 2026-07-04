// kernels/part_112_fused_triton_attn.cu
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32

__global__ void fused_triton_attn_kernel(
    float* Q, float* K, float* V, float* O,
    int batch, int heads, int seq, int dim
) {
    __shared__ float Q_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float K_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float acc_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    
    int head = bid % heads;
    int batch_idx = bid / heads;
    int head_dim = dim / heads;
    
    int q_start = (bid / heads) * seq * head_dim;
    int k_start = q_start;
    int v_start = q_start;
    
    float Q_val[4], K_val[4], V_val[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int block = 0; block < seq; block += BLOCK_SIZE) {
        for (int i = 0; i < 4; i++) {
            int q_idx = (warp * 4 + i) * 4 + lane;
            if (q_idx < seq) {
                Q_val[i] = Q[q_start + q_idx * head_dim + tid % head_dim];
            }
            int k_idx = (block + lane) * 4 + i;
            if (k_idx < seq) {
                K_val[i] = K[k_start + k_idx * head_dim + tid % head_dim];
            }
        }
        
        // Compute attention scores
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += Q_val[i] * K_val[j];
                }
                acc[i] += expf(score / sqrtf(head_dim));
            }
        }
    }
    
    for (int i = 0; i < 4; i++) {
        O[q_start + (warp * 4 + i) * head_dim + tid % head_dim] = acc[i];
    }
}