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
  } else if (j.is_number()) {
    values.push_back(j.get<float>());
  } else {
    throw std::runtime_error("Unexpected type in JSON structure: " +
                             std::string(j.type_name()));
  }
}

Tensor create_tensor_from_json(const json& j, Type type) {
  if (type == Type::kFloat) {
    try {
      std::vector<float> values;
      extract_values_from_json(j, values);
      return make_tensor(values, Shape({values.size()}));
    } catch (const json::type_error& e) {
      throw std::runtime_error("JSON type error: " + std::string(e.what()));
    } catch (const std::exception& e) {
      throw std::runtime_error("Standard exception: " + std::string(e.what()));
    } catch (...) {
      throw std::runtime_error(
          "An unknown error occurred while creating tensor from JSON.");
    }
  }
  throw std::invalid_argument("Unsupported type");
}
