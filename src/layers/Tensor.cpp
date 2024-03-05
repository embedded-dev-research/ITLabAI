#include "layers/Tensor.hpp"

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<uint8_t> Tensor::SetRightTypeValues() {
  if (type_ == kInt) {
    return std::vector<uint8_t>(shape_.count() * sizeof(int), 0);
  }
  if (type_ == kDouble) {
    return std::vector<uint8_t>(shape_.count() * sizeof(double), 0.0);
  }
}

template <typename T>
std::vector<T>* Tensor::as() {
  if (GetTypeEnum<T>() == type_) {
    return reinterpret_cast<std::vector<T>*>(&values_);
  }
  return nullptr;
}

template <typename T>
T& Tensor::operator()(const std::vector<size_t>& coords) {
  size_t s = shape_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();
  if (res_vector == nullptr) {
    throw std::invalid_argument("invalid type\n");
  }
  return (*res_vector)[s];
}  // write

template <typename T>
T Tensor::operator()(const std::vector<size_t>& coords) const {
  size_t s = shape_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();
  if (res_vector == nullptr) {
    throw std::invalid_argument("invalid type\n");
  }
  return (*res_vector)[s];
}  // read

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
  for (size_t i = 0; i < t.shape_.count(); i++) {
    out.width(5);
    out << t.values_[i] << " ";
    if ((i + 1) % t.shape_[1] == 0) out << std::endl;
  }

  return out;
}

Tensor initial_square_int_picture() {
  srand(time(nullptr));
  std::vector<size_t> initial_size = {224, 224};
  Tensor picture(initial_size, kInt);

  for (size_t h = 0; h < picture.get_size()[0]; h++) {
    for (size_t w = 0; w < picture.get_size()[1]; w++) {
      picture.operator()<int>({h, w}) = rand() % 255;
    }
  }

  return picture;
}