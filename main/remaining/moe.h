// core/nova_moe.h
#ifndef NOVA_MOE_H
#define NOVA_MOE_H

#include <cstdint>
#include <vector>
#include <unordered_map>

namespace nova {

class MoE {
public:
    MoE();
    ~MoE();

    void init(size_t num_experts, size_t top_k, size_t hidden_size, size_t intermediate_size);
    void clear();

    void set_expert_weights(size_t expert_id, const float* w1, const float* w2, size_t size);
    void set_router_weights(const float* router, size_t size);

    void forward(
        const float* input,
        float* output,
        size_t batch,
        size_t seq_len,
        size_t hidden_size
    );

    void forward_with_aux_loss(
        const float* input,
        float* output,
        float* aux_loss,
        size_t batch,
        size_t seq_len,
        size_t hidden_size
    );

    void backward(
        const float* grad_output,
        float* grad_input,
        float* grad_router,
        size_t batch,
        size_t seq_len,
        size_t hidden_size
    );

    void get_expert_counts(size_t* counts, size_t num_experts) const;
    void get_expert_weights(float* weights, size_t num_experts, size_t size) const;

    size_t get_num_experts() const { return num_experts_; }
    size_t get_top_k() const { return top_k_; }

private:
    struct Expert {
        std::vector<float> w1;
        std::vector<float> w2;
        std::vector<float> w3; // For SwiGLU
        std::vector<float> buffer;
        size_t used_count;
        float total_weight;
    };

    size_t num_experts_;
    size_t top_k_;
    size_t hidden_size_;
    size_t intermediate_size_;

    std::vector<Expert> experts_;
    std::vector<float> router_weights_;
    std::vector<float> router_scores_;
    std::vector<int> expert_indices_;
    std::vector<float> expert_weights_;
    std::vector<size_t> expert_counts_;

    void forward_expert(
        const float* input,
        float* output,
        size_t expert_id,
        size_t batch,
        size_t seq_len,
        size_t hidden_size
    );

    void compute_router_scores(
        const float* input,
        size_t batch,
        size_t seq_len,
        size_t hidden_size
    );

    void compute_expert_indices(
        size_t batch,
        size_t seq_len,
        size_t top_k
    );

    float gelu(float x) const;
    float swish(float x) const;
};

} // namespace nova