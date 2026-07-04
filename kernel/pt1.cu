// kernels/part_01_vec_add.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void vec_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void vec_add(float* a, float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}