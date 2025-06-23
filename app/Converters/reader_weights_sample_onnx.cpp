#include <iostream>

#include "Weights_Reader/reader_weights.hpp"

int main() {
  std::string json_file = MODEL_PATH_YOLO11NET_ONNX;
  json model_data = read_json(json_file);

  std::cout << "Model contains " << model_data.size()
            << " layers:" << std::endl;
  std::cout << "--------------------------------------------------"
            << std::endl;

  for (const auto& layer_data : model_data) {
    int layer_index = layer_data["index"];
    std::string layer_name = layer_data["name"];
    std::string layer_type = layer_data["type"];
    bool has_weights =
        layer_data.contains("weights") && !layer_data["weights"].empty();
    bool has_value = layer_data.contains("value");

    std::cout << "Layer " << layer_index << ": " << layer_name << " ("
              << layer_type << ")" << std::endl;

    if (layer_data.contains("attributes") &&
        !layer_data["attributes"].empty()) {
      std::cout << "  Attributes:" << std::endl;
      for (const auto& [key, value] : layer_data["attributes"].items()) {
        std::cout << "    " << key << ": ";
        if (value.is_array()) {
          std::cout << "[";
          for (const auto& v : value) {
            if (v.is_number())
              std::cout << v.get<float>() << " ";
            else if (v.is_string())
              std::cout << v.get<std::string>() << " ";
          }
          std::cout << "]";
        } else if (value.is_number()) {
          std::cout << value.get<float>();
        } else if (value.is_string()) {
          std::cout << value.get<std::string>();
        }
        std::cout << std::endl;
      }
    }

    if (has_value) {
      try {
        float value = layer_data["value"].get<float>();
        std::cout << "  Value: " << value << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "  Error processing value: " << e.what() << std::endl;
      }
    }

    if (has_weights) {
      try {
        Tensor tensor = create_tensor_from_json(layer_data, Type::kFloat);

        std::cout << "  Weights shape: " << tensor.get_shape() << std::endl;

        if (!tensor.get_bias().empty()) {
          std::cout << "  Bias size: " << tensor.get_bias().size() << std::endl;
        }
      } catch (const std::exception& e) {
        std::cerr << "  Error processing weights: " << e.what() << std::endl;
      }
    } else if (!has_value) {
      std::cout << "  No weights or value" << std::endl;
    }

    std::cout << "--------------------------------------------------"
              << std::endl;
  }

  return 0;
}