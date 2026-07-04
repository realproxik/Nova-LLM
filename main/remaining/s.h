// core/nova_speculative.h
#ifndef NOVA_SPECULATIVE_H
#define NOVA_SPECULATIVE_H

#include <cstdint>
#include <vector>
#include <functional>

namespace nova {

class SpeculativeDecoder {
public:
    using GenerateFunc = std::function<std::vector<int>(const std::vector<int>&, int)>;
    using VerifyFunc = std::function<float(const std::vector<int>&, const std::vector<int>&)>;

    SpeculativeDecoder();
    ~SpeculativeDecoder();

    void init(
        GenerateFunc draft_model,
        GenerateFunc target_model,
        VerifyFunc verify_func,
        int draft_len = 5,
        int max_tokens = 2048
    );

    void set_draft_len(int len) { draft_len_ = len; }
    void set_temperature(float temp) { temperature_ = temp; }
    void set_top_p(float p) { top_p_ = p; }
    void set_top_k(int k) { top_k_ = k; }

    std::vector<int> generate(
        const std::vector<int>& prompt,
        int max_new_tokens = 100,
        float temperature = 0.8f,
        float top_p = 0.9f,
        int top_k = 50
    );

    std::vector<int> generate_stream(
        const std::vector<int>& prompt,
        std::function<void(const std::vector<int>&)> callback,
        int max_new_tokens = 100
    );

    void reset();

private:
    GenerateFunc draft_model_;
    GenerateFunc target_model_;
    VerifyFunc verify_func_;

    int draft_len_;
    int max_tokens_;
    float temperature_;
    float top_p_;
    int top_k_;

    std::vector<int> draft_tokens_;
    std::vector<float> draft_probs_;
    std::vector<int> accepted_tokens_;
    std::vector<float> acceptance_probs_;

    bool verify_draft(
        const std::vector<int>& prompt,
        const std::vector<int>& draft,
        std::vector<int>& accepted
    );

    float compute_acceptance_prob(
        float target_prob,
        float draft_prob
    );

    int sample_token(const std::vector<float>& probs);
};

} // namespace nova