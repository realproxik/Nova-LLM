// core/nova_inference.h
#ifndef NOVA_INFERENCE_H
#define NOVA_INFERENCE_H

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace nova {

class Model;
class Tokenizer;
class KVCache;
class PagedAttention;

struct InferenceConfig {
    size_t max_batch = 32;
    size_t max_seq_len = 8192;
    size_t max_new_tokens = 2048;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    int seed = 42;
    bool do_sample = true;
    bool use_cache = true;
    bool use_flash_attention = true;
    bool use_paged_attention = true;
    bool use_speculative_decoding = false;
    int draft_len = 5;
};

struct GenerationResult {
    std::vector<int> tokens;
    std::string text;
    float avg_logprob;
    float total_time_ms;
    int tokens_per_second;
    int draft_acceptance_rate;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    void init(const InferenceConfig& config);
    void load_model(std::shared_ptr<Model> model);
    void load_tokenizer(std::shared_ptr<Tokenizer> tokenizer);

    GenerationResult generate(
        const std::vector<int>& prompt_tokens,
        int max_new_tokens = -1,
        float temperature = -1.0f,
        float top_p = -1.0f,
        int top_k = -1
    );

    GenerationResult generate(
        const std::string& prompt_text,
        int max_new_tokens = -1,
        float temperature = -1.0f,
        float top_p = -1.0f,
        int top_k = -1
    );

    void generate_stream(
        const std::vector<int>& prompt_tokens,
        std::function<void(const std::vector<int>&)> token_callback,
        int max_new_tokens = -1
    );

    void generate_stream(
        const std::string& prompt_text,
        std::function<void(const std::string&)> text_callback,
        int max_new_tokens = -1
    );

    void reset();
    void clear_cache();

    InferenceConfig get_config() const { return config_; }
    void set_config(const InferenceConfig& config) { config_ = config; }

    size_t get_memory_usage() const;
    size_t get_cache_usage() const;

private:
    InferenceConfig config_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<KVCache> kv_cache_;
    std::unique_ptr<PagedAttention> paged_attention_;

    std::vector<int> forward_pass(const std::vector<int>& tokens);
    int sample_token(const float* logits, int vocab_size);
    float compute_repetition_penalty(const std::vector<int>& tokens, int token);

    void apply_temperature(float* logits, int size, float temp);
    void apply_top_p(float* logits, int size, float p);
    void apply_top_k(float* logits, int size, int k);
    void softmax(float* logits, int size);

    int sample_greedy(const float* logits, int size);
    int sample_multinomial(const float* logits, int size);

    std::vector<int> preprocess_prompt(const std::string& text);
    std::string postprocess_result(const std::vector<int>& tokens);
};

} // namespace nova