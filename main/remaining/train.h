// training/nova_trainer.h
#ifndef NOVA_TRAINER_H
#define NOVA_TRAINER_H

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace nova {

class Model;
class Optimizer;
class DataLoader;
class LossFunction;

struct TrainerConfig {
    size_t num_epochs = 3;
    size_t batch_size = 32;
    size_t grad_accum_steps = 8;
    size_t max_steps = 100000;
    size_t eval_steps = 1000;
    size_t save_steps = 5000;
    size_t log_steps = 100;
    float learning_rate = 3e-4f;
    float weight_decay = 0.1f;
    float max_grad_norm = 1.0f;
    float warmup_ratio = 0.1f;
    std::string output_dir = "./checkpoints";
    std::string run_name = "nova_training";
    bool use_fp16 = true;
    bool use_bf16 = false;
    bool use_grad_checkpointing = false;
    bool use_mixed_precision = true;
};

struct TrainingMetrics {
    float loss;
    float learning_rate;
    float grad_norm;
    float throughput;
    size_t global_step;
    size_t epoch;
    float perplexity;
};

class Trainer {
public:
    Trainer();
    ~Trainer();

    void init(const TrainerConfig& config);
    void set_model(std::shared_ptr<Model> model);
    void set_optimizer(std::shared_ptr<Optimizer> optimizer);
    void set_data_loader(std::shared_ptr<DataLoader> data_loader);
    void set_loss_function(std::shared_ptr<LossFunction> loss_fn);

    void train();
    void train_step();
    void eval();
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);

    void set_progress_callback(std::function<void(const TrainingMetrics&)> callback);
    void set_early_stopping(float patience = 5.0f, float min_delta = 0.01f);

    TrainingMetrics get_metrics() const;
    bool is_training() const { return training_; }
    void stop_training() { training_ = false; }

private:
    TrainerConfig config_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<DataLoader> data_loader_;
    std::shared_ptr<LossFunction> loss_fn_;

    TrainingMetrics metrics_;
    std::function<void(const TrainingMetrics&)> progress_callback_;

    bool training_;
    bool early_stop_;
    float early_stop_patience_;
    float early_stop_min_delta_;
    float best_loss_;
    int patience_counter_;

    void forward_backward();
    void update_weights();
    void compute_grad_norm();
    void apply_grad_clipping();
    void update_learning_rate();
    void log_metrics();

    std::vector<float> convert_to_fp16(const std::vector<float>& data);
    std::vector<float> convert_to_bf16(const std::vector<float>& data);
};

} // namespace nova