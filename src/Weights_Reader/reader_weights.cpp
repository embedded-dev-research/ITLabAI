#include "Weights_Reader/reader_weights.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

// Функция для чтения JSON файла
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

// Функция для извлечения значений из JSON
void extract_values_from_json(const json& j, std::vector<float>& values) {
  if (j.is_array()) {
    for (const auto& item : j) {
      extract_values_from_json(item, values);
    }
  } else if (j.is_number()) {  // Изменено на j.is_number()
    values.push_back(j.get<float>());
  } else if (!j.is_null()) {
    throw std::runtime_error("Unexpected type in JSON structure: " +
                             std::string(j.type_name()));
  }
}

// Функция для определения формы из JSON
void parse_json_shape(const json& j, std::vector<size_t>& shape,
                      size_t dim) {
  if (dim == 0) {
    // Игнорируем первую размерность и переходим к следующей вложенности
    if (j.is_array() && !j.empty()) {
      parse_json_shape(j.front(), shape, dim + 1);
    }
  } else {
    if (j.is_array()) {
      if (shape.size() <= dim - 1) {
        shape.push_back(j.size());
      } else if (shape[dim - 1] != j.size()) {
        throw std::runtime_error("Inconsistent array size at dimension " +
                                 std::to_string(dim - 1));
      }
      if (!j.empty()) {
        parse_json_shape(j.front(), shape, dim + 1);
      }
    } else if (!j.is_number() && !j.is_null()) {  // Изменено на j.is_number()
      throw std::runtime_error("Unexpected type in JSON structure: " +
                               std::string(j.type_name()));
    }
  }
}

void extract_bias_from_json(const json& j, std::vector<float>& bias) {
  if (j.is_array()) {
    // Проверяем, что входные данные представляют собой массив
    auto& last_element = j.back();
    if (last_element.is_array()) {
      // Если последний элемент массива также является массивом, это может быть
      // bias
      for (const auto& item : last_element) {
        if (item.is_number()) {
          bias.push_back(item.get<float>());
        } else {
          throw std::runtime_error("Unexpected type in bias array: " +
                                   std::string(item.type_name()));
        }
      }
    } else {
      throw std::runtime_error("Last element should be an array (bias).");
    }
  } else {
    throw std::runtime_error("Input JSON structure should be an array.");
  }
}

// Функция для создания тензора из JSON
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
    size_t expected_size = 1;
    for (const auto& dim : shape) {
      std::cout << dim << " ";
      expected_size *= dim;
    }
    std::cout << std::endl;

    // Обработка пустых слоев
    if (expected_size == 1 && shape.empty()) {
      expected_size = 0;
    }

    std::cout << "Expected size: " << expected_size << std::endl;

    if (vals.size() != expected_size) {
      throw std::runtime_error(
          "Incorrect vector size given to Tensor. Extracted values size: " +
          std::to_string(vals.size()) +
          ", Expected size: " + std::to_string(expected_size));
    }

    Shape sh(shape);
    return make_tensor<float>(vals, sh);
  }
  throw std::invalid_argument("Unsupported type or invalid JSON format");
}
