// kernels/part_87_pipeline_parallel_sendrecv.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void pipeline_parallel_sendrecv_kernel(
    float* send_buffer, float* recv_buffer,
    int send_stage, int recv_stage,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * seq * dim) return;
    
    recv_buffer[recv_stage * batch * seq * dim + idx] = 
        send_buffer[send_stage * batch * seq * dim + idx];
}