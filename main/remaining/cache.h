// core/nova_kv_cache.h
#ifndef NOVA_KV_CACHE_H
#define NOVA_KV_CACHE_H

#include <cstdint>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace nova {

struct KVBlock {
    uint64_t page_id;
    uint64_t token_count;
    uint64_t capacity;
    float* k_data;
    float* v_data;
    bool active;
    uint64_t last_used;
};

class KVCache {
public:
    KVCache();
    ~KVCache();

    void init(size_t num_heads, size_t head_dim, size_t max_seq, size_t max_batch);
    void clear();

    void allocate_block(uint64_t page_id, size_t seq_len);
    void free_block(uint64_t page_id);

    float* get_k(uint64_t page_id, size_t pos);
    float* get_v(uint64_t page_id, size_t pos);

    void append(uint64_t page_id, const float* k, const float* v, size_t seq_len);
    void copy_from(uint64_t dst_page, uint64_t src_page, size_t start, size_t len);
    void copy_to(uint64_t dst_page, uint64_t src_page, size_t start, size_t len);

    size_t get_total_pages() const;
    size_t get_active_pages() const;
    size_t get_memory_usage() const;

    void evict_lru();
    void evict_fifo();
    void evict_lfu();

private:
    struct CachePage {
        uint64_t id;
        std::vector<float> k_cache;
        std::vector<float> v_cache;
        uint64_t seq_len;
        uint64_t max_seq;
        uint64_t last_used;
        uint64_t use_count;
        bool active;
    };

    std::unordered_map<uint64_t, CachePage> pages_;
    std::mutex mutex_;
    size_t num_heads_;
    size_t head_dim_;
    size_t max_seq_;
    size_t max_batch_;
    size_t total_memory_;
    size_t used_memory_;

    uint64_t get_page_size() const;
};

} // namespace nova
