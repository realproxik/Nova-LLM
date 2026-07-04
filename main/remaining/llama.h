// models/nova_llama.h
#ifndef NOVA_LLAMA_H
#define NOVA_LLAMA_H

#include <cstdint>
#include <vector>
#include <memory>

namespace nova {

class Model;

struct LLaMAConfig {
    size_t vocab_size = 32000;
    size_t hidden_size = 4096;
    size_t num_layers = 32;
    size_t num_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    size_t max_seq_len = 8192;
    size_t intermediate_size = 11008;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool use_bias = false;
    bool use_mlp_bias = false;
    float dropout = 0.0f;
    bool use_moe = false;
    size_t num_experts = 8;
    size_t top_k_experts = 2;
};

class LLaMAModel {
public:
    LLaMAModel();
    explicit LLaMAModel(const LLaMAConfig& config);
    ~LLaMAModel();

    void init(const LLaMAConfig& config);
    void forward(const float* input, float* output, size_t batch, size_t seq_len);
    void backward(const float* grad_output, float* grad_input);

    void set_weights(const std::vector<float>& weights);
    std::vector<float> get_weights() const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t get_vocab_size() const { return config_.vocab_size; }
    size_t get_hidden_size() const { return config_.hidden_size; }
    size_t get_num_layers() const { return config_.num_layers; }

private:
    struct LayerWeights {
        std::vector<float> attn_q_weight;
        std::vector<float> attn_k_weight;
        std::vector<float> attn_v_weight;
        std::vector<float> attn_o_weight;
        std::vector<float> mlp_gate_weight;
        std::vector<float> mlp_up_weight;
        std::vector<float> mlp_down_weight;
        std::vector<float> attn_norm_weight;
        std::vector<float> mlp_norm_weight;
    };

    LLaMAConfig config_;
    std::vector<LayerWeights> layers_;
    std::vector<float> embed_table_;
    std::vector<float> final_norm_;
    std::vector<float> lm_head_;

    std::vector<float> activations_;
    std::vector<float> gradients_;

    void rms_norm(const float* input, float* output, const float* weight, size_t size, float eps);
    void rope_embed(float* q, float* k, size_t seq_len, size_t head_dim, float theta);
    void swiglu_mlp(const float* input, float* output,
                    const float* gate, const float* up, const float* down,
                    size_t hidden, size_t intermediate);
    void attention(const float* q, const float* k, const float* v, float* output,
                   size_t batch, size_t seq_len, size_t num_heads, size_t head_dim,
                   bool causal);
};

} // namespace nova