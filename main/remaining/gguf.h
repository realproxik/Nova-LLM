// core/nova_gguf.h
#ifndef NOVA_GGUF_H
#define NOVA_GGUF_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace nova {

struct GGUFValue {
    enum Type {
        UINT8, INT8, UINT16, INT16, UINT32, INT32,
        FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64
    };
    Type type;
    std::vector<uint8_t> data;
};

struct GGUFMetadata {
    std::string magic;                      // "GGUF"
    uint32_t version;                       // 3
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::map<std::string, GGUFValue> metadata;
    std::vector<std::string> tensor_names;
    std::vector<uint64_t> tensor_offsets;
    std::vector<uint64_t> tensor_sizes;
    std::vector<uint32_t> tensor_types;
};

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    bool load(const std::string& path);
    bool load_header();
    bool load_metadata();
    bool load_tensors();
    bool load_tensor_data(const std::string& name, float* buffer, size_t size);

    GGUFMetadata get_metadata() const { return metadata_; }
    size_t get_tensor_count() const { return tensor_count_; }
    size_t get_total_size() const { return total_size_; }
    bool has_tensor(const std::string& name) const;

    std::vector<std::string> get_tensor_names() const;
    std::vector<size_t> get_tensor_shapes(const std::string& name) const;
    std::string get_chat_template() const;

private:
    std::string path_;
    std::ifstream file_;
    GGUFMetadata metadata_;
    size_t tensor_count_;
    size_t total_size_;
    std::map<std::string, std::pair<size_t, size_t>> tensor_index_; // name -> (offset, size)

    bool read_magic();
    bool read_version();
    bool read_metadata_kv_count();
    bool read_metadata_value(GGUFValue& value);
    bool read_tensor_info();
    void read_string(std::string& str);
    void read_data(void* buffer, size_t size);
    template<typename T> T read_value();
};

} // namespace nova