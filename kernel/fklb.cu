// kernels/part_106_fused_loss_backward.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_loss_backward_kernel(
    float* logits, int* targets,
    float* grad_logits,
    int batch, int vocab
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int b = bid;
    int target = targets[b];
    
    float max_val = logits[b * vocab];
    for (int i = 1; i < vocab; i++) {
        if (logits[b * vocab + i] > max_val) max_val = logits[b * vocab + i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        sum += expf(logits[b * vocab + i] - max_val);
    }
    
    for (int i = tid; i < vocab; i += blockDim.x) {
        float prob = expf(logits[b * vocab + i] - max_val) / sum;
        grad_logits[b * vocab + i] = prob - (i == target ? 1.0f : 0.0f);
    }
}