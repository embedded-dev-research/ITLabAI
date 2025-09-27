#include "Weights_Reader/reader_weights.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

namespace it_lab_ai {

using json = nlohmann::json;

json read_json(const std::string& filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + filename);
  }

  json model_data;
  try {
    ifs >> model_data;
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

void parse_json_shape(const json& j, std::vector<size_t>& shape,
                      size_t dim = 0) {
  if (!j.is_array()) {
    if (dim == 0) shape.push_back(0);
    return;
  }

  if (shape.size() <= dim) {
    shape.push_back(j.size());
  }

  if (!j.empty()) {
    parse_json_shape(j[0], shape, dim + 1);
  }
}

Tensor create_tensor_from_json(const json& layer_data, Type type) {
  if (type != Type::kFloat) {
    throw std::invalid_argument("Only float type is supported");
  }

  std::vector<float> weights;
  if (layer_data.contains("weights") && !layer_data["weights"].empty()) {
    extract_values_from_json(layer_data["weights"], weights);
  }

  std::vector<float> bias;
  if (layer_data.contains("bias") && !layer_data["bias"].empty()) {
    extract_values_from_json(layer_data["bias"], bias);
  }

  std::vector<size_t> shape;
  if (layer_data.contains("weights")) {
    parse_json_shape(layer_data["weights"], shape);
  }

  return make_tensor<float>(weights, Shape(shape), bias);
}

}  // namespace it_lab_ai
