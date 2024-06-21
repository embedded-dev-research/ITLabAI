#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include "layers/Tensor.hpp"

using json = nlohmann::json;
using namespace itlab_2023;

json read_json(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open JSON file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  if (size == 0) {
    std::cerr << "JSON file is empty: " << filename << std::endl;
    ifs.close();
    exit(EXIT_FAILURE);
  }
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  ifs.read(buffer.data(), size);
  ifs.close();

  json model_data;
  try {
    model_data = json::parse(buffer.begin(), buffer.end());
  } catch (const json::parse_error& e) {
    std::cerr << "JSON parse error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  } catch (const std::exception& e) {
    std::cerr << "Standard exception: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "An unknown error occurred while parsing JSON." << std::endl;
    exit(EXIT_FAILURE);
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
  } else {
    std::cerr << "Unexpected type in JSON structure: " << j.type_name()
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

Tensor create_tensor_from_json(const json& j, Type type) {
  if (type == Type::kFloat) {
    try {
      std::vector<float> values;
      extract_values_from_json(j, values);
      return make_tensor(values, Shape({values.size()}));
    } catch (const json::type_error& e) {
      std::cerr << "JSON type error: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    } catch (const std::exception& e) {
      std::cerr << "Standard exception: " << e.what() << std::endl;
      exit(EXIT_FAILURE);
    } catch (...) {
      std::cerr << "An unknown error occurred while creating tensor from JSON."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  throw std::invalid_argument("Unsupported type");
}

int main() {
  std::string json_file = "model_data_alexnet_1.json";
  json model_data = read_json(json_file);

  for (auto& layer : model_data.items()) {
    std::string layer_name = layer.key();
    std::cout << "Layer: " << layer_name << std::endl;

    for (const auto& weight : layer.value()) {
      try {
        Tensor tensor = create_tensor_from_json(weight, Type::kFloat);
        std::cout << tensor << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "Error processing layer " << layer_name << ": " << e.what()
                  << std::endl;
      }
    }
  }

  return 0;
}
