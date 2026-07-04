// kernels/part_83_paged_attn_page_lookup.cu
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void paged_attn_page_lookup_kernel(
    int* page_table, int* page_indices,
    int* physical_pages, int num_pages,
    int batch, int seq, int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * seq) return;
    
    int b = idx / seq;
    int s = idx % seq;
    int page = s / block_size;
    int offset = s % block_size;
    
    int logical_page = page_indices[b * num_pages + page];
    physical_pages[b * seq + s] = page_table[logical_page] * block_size + offset;
}