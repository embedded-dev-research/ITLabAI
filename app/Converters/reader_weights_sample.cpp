#include <iostream>

#include "Weights_Reader/reader_weights.hpp"

int main() {
  std::string json_file = MODEL_PATH_H5;
  it_lab_ai::json model_data = it_lab_ai::read_json(json_file);

  for (const auto& layer_data : model_data) {
    int layer_index = layer_data["index"];
    std::string layer_name = layer_data["name"];
    std::string layer_type = layer_data["type"];

    std::cout << "Layer " << layer_index << " (" << layer_type << ", "
              << layer_name << "):" << std::endl;

    try {
      it_lab_ai::Tensor tensor = it_lab_ai::create_tensor_from_json(
          layer_data, it_lab_ai::Type::kFloat);
      // std::cout << tensor << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error processing layer " << layer_name << ": " << e.what()
                << std::endl;
    }
  }

  return 0;
}