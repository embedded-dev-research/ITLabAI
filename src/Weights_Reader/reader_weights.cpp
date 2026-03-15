#include "Weights_Reader/reader_weights.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace it_lab_ai {

using json = nlohmann::json;

json read_json(const std::string& filename) {
#ifdef _WIN32
  HANDLE file = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (file == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  DWORD size = GetFileSize(file, NULL);
  if (size == 0) {
    CloseHandle(file);
    return json{};
  }

  HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
  char* data = (char*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);

  json result = json::parse(data, data + size);

  UnmapViewOfFile(data);
  CloseHandle(mapping);
  CloseHandle(file);
  return result;

#else
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  struct stat sb {};
  fstat(fd, &sb);

  if (sb.st_size == 0) {
    close(fd);
    return json{};
  }

  char* data = static_cast<char*>(
      mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  json result = json::parse(data, data + sb.st_size);

  munmap(data, sb.st_size);
  close(fd);
  return result;
#endif
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
    if (dim == 0) {
      shape.push_back(0);
    }
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
