// tokenizer/nova_bpe.h
#ifndef NOVA_BPE_H
#define NOVA_BPE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace nova {

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    void init(const std::string& vocab_path, const std::string& merges_path);
    void train(const std::vector<std::string>& texts, size_t vocab_size);
    void save(const std::string& path) const;
    void load(const std::string& path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    void add_special_token(const std::string& token);
    void set_special_tokens(const std::string& bos, const std::string& eos,
                            const std::string& pad, const std::string& unk);

    size_t vocab_size() const { return vocab_.size(); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    int unk_id() const { return unk_id_; }

private:
    struct Pair {
        int first;
        int second;

        bool operator==(const Pair& other) const {
            return first == other.first && second == other.second;
        }

        struct Hash {
            size_t operator()(const Pair& p) const {
                return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
            }
        };
    };

    std::unordered_map<int, std::string> vocab_;
    std::unordered_map<std::string, int> reverse_vocab_;
    std::unordered_map<Pair, int, Pair::Hash> merges_;
    std::unordered_map<int, std::unordered_set<int>> merge_rules_;

    int bos_id_;
    int eos_id_;
    int pad_id_;
    int unk_id_;

    std::vector<std::string> get_words(const std::string& text) const;
    std::vector<int> word_to_tokens(const std::string& word) const;
    std::string tokens_to_word(const std::vector<int>& tokens) const;

    void build_vocab(const std::vector<std::string>& texts);
    void compute_pair_frequencies(const std::vector<std::vector<int>>& tokenized);
    void merge_pair(int a, int b, int new_id);
    void add_merge_rule(int a, int b, int new_id);

    std::vector<int> bpe_merge(const std::vector<int>& tokens) const;
};

} // namespace nova