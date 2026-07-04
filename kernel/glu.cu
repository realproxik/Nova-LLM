// kernels/part_58_mlp_glu.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mlp_glu_kernel(
    float* input, float* w1, float* w2, float* w3, float* output,
    int batch, int seq, int hidden, int intermediate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq;
    
    if (idx >= total) return;
    
    int b = idx / seq;
    int s = idx % seq;
    
    float* in = input + (b * seq + s) * hidden;
    float* out = output + (b * seq + s) * hidden;
    
    float temp[4096 * 4];
    
    for (int i = 0; i < intermediate; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += in[j] * w1[i * hidden + j];
        }
        temp[i] = sum;
    }
    
    for (int i = 0; i < intermediate; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += in[j] * w3[i * hidden + j];
        }
        float gate = 1.0f / (1.0f + expf(-sum));
        temp[i] *= gate;
    }
    
    for (int i = 0; i < hidden; i++) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate; j++) {
            sum += temp[j] * w2[i * intermediate + j];
        }
        out[i] = sum;
    }
}