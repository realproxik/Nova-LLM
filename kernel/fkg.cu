// kernels/part_105_fused_grad_accum_adamw.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_grad_accum_adamw_kernel(
    float* grad, float* accum_grad,
    float* param, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    int step, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float g = grad[idx];
    accum_grad[idx] += g;
    
    float p = param[idx];
    float ag = accum_grad[idx];
    float m_val = m[idx];
    float v_val = v[idx];
    
    m_val = beta1 * m_val + (1.0f - beta1) * ag;
    v_val = beta2 * v_val + (1.0f - beta2) * ag * ag;
    
    m[idx] = m_val;
    v[idx] = v_val;
    
    float m_hat = m_val / (1.0f - powf(beta1, step));
    float v_hat = v_val / (1.0f - powf(beta2, step));
    
    param[idx] = p - lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
    accum_grad[idx] = 0.0f;
}