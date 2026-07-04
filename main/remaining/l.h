// training/nova_loss.h
#ifndef NOVA_LOSS_H
#define NOVA_LOSS_H

#include <vector>
#include <cstdint>

namespace nova {

class LossFunction {
public:
    virtual ~LossFunction() = default;

    virtual float compute(const std::vector<float>& predictions, const std::vector<float>& targets) = 0;
    virtual std::vector<float> gradient() = 0;
    virtual void reset() = 0;
};

class CrossEntropyLoss : public LossFunction {
public:
    CrossEntropyLoss();
    explicit CrossEntropyLoss(float label_smoothing = 0.0f, float ignore_index = -100.0f);
    ~CrossEntropyLoss();

    float compute(const std::vector<float>& predictions, const std::vector<float>& targets) override;
    std::vector<float> gradient() override;
    void reset() override;

    void set_label_smoothing(float smoothing) { label_smoothing_ = smoothing; }
    void set_ignore_index(float index) { ignore_index_ = index; }

private:
    std::vector<float> grad_;
    float loss_;
    float label_smoothing_;
    float ignore_index_;
    size_t batch_size_;
    size_t vocab_size_;

    void softmax(const std::vector<float>& logits, std::vector<float>& probs);
    void compute_gradient(const std::vector<float>& probs, const std::vector<float>& targets);
};

class MSELoss : public LossFunction {
public:
    MSELoss();
    explicit MSELoss(float reduction = 0.0f);
    ~MSELoss();

    float compute(const std::vector<float>& predictions, const std::vector<float>& targets) override;
    std::vector<float> gradient() override;
    void reset() override;

private:
    std::vector<float> grad_;
    float loss_;
    float reduction_;
    size_t size_;
};

class KLDivergenceLoss : public LossFunction {
public:
    KLDivergenceLoss();
    explicit KLDivergenceLoss(bool log_target = false);
    ~KLDivergenceLoss();

    float compute(const std::vector<float>& predictions, const std::vector<float>& targets) override;
    std::vector<float> gradient() override;
    void reset() override;

private:
    std::vector<float> grad_;
    float loss_;
    bool log_target_;
};

} // namespace nova