// kernels/part_24_quant_int8.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void quantize_int8_kernel(float* input, int8_t* output, float* scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }
    
    float scale_val = 127.0f / max_val;
    *scale = scale_val;
    
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = input[i] * scale_val;
        val = fminf(127.0f, fmaxf(-127.0f, val));
        output[i] = (int8_t)roundf(val);
    }
}

__global__ void quantize_int8_per_tensor_kernel(float* input, int8_t* output, float* scale, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    int start = bid * size;
    int end = min(start + size, size);
    
    float max_val = 0.0f;
    for (int i = start + tid; i < end; i += blockDim.x) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }
    shared_mem[tid] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        scale[bid] = 127.0f / shared_mem[0];
    }
    __syncthreads();
    
    float scale_val = scale[bid];
    for (int i = start + tid; i < end; i += blockDim.x) {
        float val = input[i] * scale_val;
        val = fminf(127.0f, fmaxf(-127.0f, val));
        output[i] = (int8_t)roundf(val);
    }
}