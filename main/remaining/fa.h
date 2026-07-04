// core/nova_flash_attention.h
#ifndef NOVA_FLASH_ATTENTION_H
#define NOVA_FLASH_ATTENTION_H

#include <cstdint>
#include <vector>

namespace nova {

class FlashAttention {
public:
    FlashAttention();
    ~FlashAttention();

    void init(size_t num_heads, size_t head_dim, size_t block_size = 128);
    void forward(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        size_t batch,
        size_t seq_len,
        bool causal = true,
        float scale = -1.0f
    );

    void backward(
        const float* dO,
        const float* Q,
        const float* K,
        const float* V,
        const float* O,
        float* dQ,
        float* dK,
        float* dV,
        size_t batch,
        size_t seq_len
    );

private:
    void forward_kernel(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        size_t batch,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim,
        size_t block_size,
        bool causal,
        float scale
    );

    void backward_kernel(
        const float* dO,
        const float* Q,
        const float* K,
        const float* V,
        const float* O,
        float* dQ,
        float* dK,
        float* dV,
        size_t batch,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim,
        size_t block_size
    );

    size_t num_heads_;
    size_t head_dim_;
    size_t block_size_;
};

} // namespace nova