#include <fstream>
#include <iostream>

#include "Weights_Reader/reader_weights.hpp"

json read_json(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + filename);
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  if (size == 0) {
    ifs.close();
    throw std::runtime_error("JSON file is empty: " + filename);
  }
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  ifs.read(buffer.data(), size);
  ifs.close();

  json model_data;
  try {
    model_data = json::parse(buffer.begin(), buffer.end());
  } catch (const json::parse_error& e) {
    throw std::runtime_error("JSON parse error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    throw std::runtime_error("Standard exception: " + std::string(e.what()));
  } catch (...) {
    throw std::runtime_error("An unknown error occurred while parsing JSON.");
  }

  return model_data;
}


void extract_values_from_json(const json& j, std::vector<float>& values) {
  if (j.is_array()) {
    for (const auto& item : j) {
      extract_values_from_json(item, values);
    }
  } else if (j.is_number_float()) {
    values.push_back(j.get<float>());
  } else if (!j.is_null()) {
    throw std::runtime_error("Unexpected type in JSON structure: " +
                             std::string(j.type_name()));
  }
}


void parse_json_shape(const json& j, std::vector<size_t>& shape,
                      size_t dim) {
  if (j.is_array()) {
    if (shape.size() <= dim) {
      shape.push_back(j.size());
    } else if (shape[dim] != j.size()) {
      throw std::runtime_error("Inconsistent array size at dimension " +
                               std::to_string(dim));
    }
    if (!j.empty()) {
      parse_json_shape(j.front(), shape, dim + 1);
    }
  } else if (!j.is_number_float() && !j.is_null()) {
    throw std::runtime_error("Unexpected type in JSON structure: " +
                             std::string(j.type_name()));
  }
}

Tensor create_tensor_from_json(const json& j, Type type) {

    if (type == Type::kFloat) {
      std::vector<float> vals;
      std::vector<size_t> shape;

      // Извлечение значений из JSON
      extract_values_from_json(j, vals);
      std::cout << "Extracted values size: " << vals.size() << std::endl;

      // Определение формы тензора
      parse_json_shape(j, shape);
      std::cout << "Parsed shape: ";
      for (const auto& dim : shape) {
        std::cout << dim << " ";
      }
      std::cout << std::endl;

      Shape sh(shape);
      return make_tensor<float>(vals, sh);
    }
    throw std::invalid_argument("Unsupported type or invalid JSON format");

}
