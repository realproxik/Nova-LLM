// training/nova_data_loader.h
#ifndef NOVA_DATA_LOADER_H
#define NOVA_DATA_LOADER_H

#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <filesystem>

namespace nova {

struct Batch {
    std::vector<float> inputs;
    std::vector<float> targets;
    std::vector<size_t> indices;
    size_t batch_size;
};

struct Dataset {
    std::vector<std::vector<int>> tokens;
    std::vector<int> labels;
    size_t size;
    size_t max_len;
};

class DataLoader {
public:
    DataLoader();
    ~DataLoader();

    void init(const std::string& data_path, size_t batch_size, size_t seq_len, bool shuffle = true);
    void load_data(const std::string& path);

    Batch next_batch();
    Batch current_batch() const { return current_batch_; }
    bool has_next() const;

    void reset();
    void shuffle();

    std::vector<Batch> get_eval_data(size_t num_batches = 10);

    size_t get_batch_size() const { return batch_size_; }
    size_t get_num_batches() const { return num_batches_; }
    size_t get_dataset_size() const { return dataset_.size; }

private:
    struct InternalBatch {
        std::vector<std::vector<int>> token_batch;
        std::vector<int> label_batch;
    };

    Dataset dataset_;
    Batch current_batch_;

    size_t batch_size_;
    size_t seq_len_;
    size_t num_batches_;
    size_t current_index_;
    bool shuffle_;

    std::random_device rd_;
    std::mt19937 gen_;

    InternalBatch get_raw_batch();
    Batch convert_batch(const InternalBatch& raw);
    void pad_batch(InternalBatch& raw);
};

class Tokenizer;

class TextDataset {
public:
    TextDataset();
    ~TextDataset();

    void load_file(const std::string& path);
    void load_directory(const std::string& path);

    void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer);
    void set_max_len(size_t max_len) { max_len_ = max_len; }

    std::vector<int> encode_text(const std::string& text);
    std::string decode_tokens(const std::vector<int>& tokens);

    size_t size() const { return texts_.size(); }

private:
    std::vector<std::string> texts_;
    std::shared_ptr<Tokenizer> tokenizer_;
    size_t max_len_;
};

} // namespace nova