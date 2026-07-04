// training/nova_optimizer.h
#ifndef NOVA_OPTIMIZER_H
#define NOVA_OPTIMIZER_H

#include <cstdint>
#include <vector>
#include <string>

namespace nova {

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void init() = 0;
    virtual void step() = 0;
    virtual void zero_grad() = 0;

    virtual void add_parameter(float* param, float* grad, size_t size) = 0;
    virtual void set_learning_rate(float lr) = 0;
    virtual float get_learning_rate() const = 0;

    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
};

class AdamWOptimizer : public Optimizer {
public:
    AdamWOptimizer();
    explicit AdamWOptimizer(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                            float eps = 1e-8f, float weight_decay = 0.01f);
    ~AdamWOptimizer();

    void init() override;
    void step() override;
    void zero_grad() override;

    void add_parameter(float* param, float* grad, size_t size) override;
    void set_learning_rate(float lr) override { lr_ = lr; }
    float get_learning_rate() const override { return lr_; }

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    struct ParameterState {
        float* param;
        float* grad;
        size_t size;
        std::vector<float> m;  // First moment
        std::vector<float> v;  // Second moment
    };

    std::vector<ParameterState> params_;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    size_t step_;

    bool initialized_;
};

class SGD : public Optimizer {
public:
    SGD();
    explicit SGD(float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    ~SGD();

    void init() override;
    void step() override;
    void zero_grad() override;

    void add_parameter(float* param, float* grad, size_t size) override;
    void set_learning_rate(float lr) override { lr_ = lr; }
    float get_learning_rate() const override { return lr_; }

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    struct ParameterState {
        float* param;
        float* grad;
        size_t size;
        std::vector<float> velocity;
    };

    std::vector<ParameterState> params_;
    float lr_;
    float momentum_;
    float weight_decay_;
    bool initialized_;
};

} // namespace nova