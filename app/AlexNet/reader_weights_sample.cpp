#include <iostream>

#include "Weights_Reader/reader_weights.hpp"

int main() {
  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);

  for (const auto& layer_data : model_data) {
    int layer_index = layer_data["index"];
    std::string layer_name = layer_data["name"];
    std::string layer_type = layer_data["type"];

    std::cout << "Layer " << layer_index << " (" << layer_type << ", "
              << layer_name << "):" << std::endl;

    try {
      Tensor tensor =
          create_tensor_from_json(layer_data["weights"], Type::kFloat);
      // std::cout << tensor << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error processing layer " << layer_name << ": " << e.what()
                << std::endl;
    }
  }

  return 0;
}