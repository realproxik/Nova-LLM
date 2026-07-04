// kernels/part_77_speculative_sampling.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void speculative_sampling_kernel(
    float* draft_logits, float* target_logits,
    int* draft_tokens, int* accepted_tokens,
    int batch, int vocab, int draft_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * draft_len) return;
    
    int b = idx / draft_len;
    int d = idx % draft_len;
    int token = draft_tokens[idx];
    
    float draft_prob = expf(draft_logits[idx * vocab + token]);
    float target_prob = expf(target_logits[idx * vocab + token]);
    
    float acceptance = fminf(1.0f, target_prob / (draft_prob + 1e-7f));
    float r = (float)rand() / RAND_MAX;
    
    if (r < acceptance) {
        accepted_tokens[idx] = 1;
    } else {
        accepted_tokens[idx] = 0;
    }
}