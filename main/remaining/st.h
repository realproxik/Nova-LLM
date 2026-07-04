// core/nova_safetensors.h
#ifndef NOVA_SAFETENSORS_H
#define NOVA_SAFETENSORS_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace nova {

struct SafetensorInfo {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;
    size_t offset;
    size_t size;
};

class SafetensorsLoader {
public:
    SafetensorsLoader();
    ~SafetensorsLoader();

    bool load(const std::string& path);
    bool load_header();
    bool load_tensor_data(const std::string& name, float* buffer, size_t size);

    std::vector<std::string> get_tensor_names() const;
    std::vector<size_t> get_tensor_shape(const std::string& name) const;
    size_t get_tensor_size(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

private:
    std::string path_;
    std::ifstream file_;
    std::map<std::string, SafetensorInfo> tensors_;
    size_t header_size_;
    size_t data_offset_;

    bool parse_header(const std::string& header_json);
    size_t dtype_size(const std::string& dtype) const;
};

} // namespace nova