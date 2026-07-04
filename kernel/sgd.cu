// kernels/part_65_sgd_step.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sgd_step_kernel(
    float* param, float* grad,
    float lr, float momentum, float wd,
    float* velocity,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float p = param[idx];
    float g = grad[idx] + wd * p;
    
    if (momentum > 0.0f) {
        float v = velocity[idx];
        v = momentum * v + lr * g;
        velocity[idx] = v;
        param[idx] = p - v;
    } else {
        param[idx] = p - lr * g;
    }
}