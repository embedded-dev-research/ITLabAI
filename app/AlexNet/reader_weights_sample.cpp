#include <iostream>

#include "Weights_Reader/reader_weights.hpp"

int main() {
  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);

  for (const auto& layer : model_data.items()) {
    const std::string& layer_name = layer.key();
    std::cout << "Layer: " << layer_name << std::endl;

    try {
      Tensor tensor = create_tensor_from_json(layer.value(), Type::kFloat);
      std::cout << tensor << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error processing layer " << layer_name << ": " << e.what()
                << std::endl;
    }
  }

  return 0;
}