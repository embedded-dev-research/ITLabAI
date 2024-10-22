#include "Weights_Reader/reader_weights.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

json read_json(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + filename);
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  ifs.read(buffer.data(), size);
  ifs.close();

  json model_data;
  try {
    model_data = json::parse(buffer.begin(), buffer.end());
  } catch (const json::parse_error& e) {
    throw std::runtime_error("JSON parse error: " + std::string(e.what()));
  }

  return model_data;
}

void extract_values_from_json(const json& j, std::vector<float>& values) {
  if (j.is_array()) {
    for (const auto& item : j) {
      extract_values_from_json(item, values);
    }
  } else if (j.is_number()) {
    values.push_back(j.get<float>());
  }
}

void extract_values_without_bias(const json& j, std::vector<float>& values) {
  std::vector<float> temp_values;
  extract_values_from_json(j, temp_values);
  size_t bias_size = 0;
  if (j.is_array() && !j.empty() && j.back().is_array()) {
    bias_size = j.back().size();
  }
  std::cout << "Bias size: " << bias_size << std::endl;
  if (temp_values.size() >= bias_size) {
    values.assign(temp_values.begin(), temp_values.end() - bias_size);
  }
}

void parse_json_shape(const json& j, std::vector<size_t>& shape, size_t dim) {
  if (dim == 0) {
    if (j.is_array()) {
      if (j.empty()) {
        shape.push_back(0);
        return;
      }
      parse_json_shape(j.front(), shape, dim + 1);
    } else {
      shape.push_back(0);
    }
  } else {
    if (j.is_array()) {
      if (shape.size() <= dim - 1) {
        shape.push_back(j.size());
      }
      if (!j.empty()) {
        parse_json_shape(j.front(), shape, dim + 1);
      }
    }
  }
}

void extract_bias_from_json(const json& j, std::vector<float>& bias) {
  if (j.is_array()) {
    if (!j.empty() && j.back().is_array()) {
      for (const auto& item : j.back()) {
        if (item.is_number()) {
          bias.push_back(item.get<float>());
        }
      }
    }
  }
}

Tensor create_tensor_from_json(const json& j, Type type) {
  if (type == Type::kFloat) {
    std::vector<float> vals;
    std::vector<size_t> shape;
    std::vector<float> bias;
    extract_values_without_bias(j, vals);
    std::cout << "Extracted values size: " << vals.size() << std::endl;

    parse_json_shape(j, shape);
    std::cout << "Parsed shape: ";
    size_t expected_size = 1;
    for (const auto& dim : shape) {
      std::cout << dim << " ";
      expected_size *= dim;
    }
    std::cout << std::endl;

    if (expected_size == 1 && shape.empty()) {
      expected_size = 0;
    }
    extract_bias_from_json(j, bias);
    std::cout << "Extracted bias size: " << bias.size() << std::endl;
    Shape sh(shape);
    return make_tensor<float>(vals, sh, bias);
  }
  throw std::invalid_argument("Unsupported type or invalid JSON format");
}
