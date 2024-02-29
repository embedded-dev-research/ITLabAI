#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <stdexcept>

#include "layers/Layer.hpp"
#include "Tensor.hpp"

std::vector<uint8_t> Tensor::SetRightTypeValues() {
  if (type_ == INT) {
    return std::vector<uint8_t>(size_.count() * sizeof(int), 0);
  }
  if (type_ == DOUBLE) {
    return std::vector<uint8_t>(size_.count() * sizeof(double), 0.0);
  }
}

template <typename T>
std::vector<T>* Tensor::as() {
  if (GetTypeEnum<T>() == type_){
    return reinterpret_cast<std::vector<T>*>(&values_);
  }
  return nullptr;
}

template <typename T>
double& Tensor::operator()(const std::vector<size_t>& coords) {
  size_t s = size_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();
  if (res_vector == nullptr) {
    throw std::exception("invalid type\n");
  }
  return (*res_vector)[s];
} // write

template <typename T>
double Tensor::operator()(const std::vector<size_t>& coords) const {
  size_t s = size_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();
  if (res_vector == nullptr) {
    throw std::exception("invalid type\n");
  }
  return (*res_vector)[s];
} // read


std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    for (int i = 0; i < t.size_.count(); i++) {
      out.width(5);
      out << t.values_[i] << " ";
      if ((i + 1) % t.size_[1] == 0)
        out << std::endl;
    }

    return out;
}

Tensor initial_square_int_picture() {
    srand(time(nullptr));
    std::vector<size_t> initial_size = {224, 224};
    Tensor picture(initial_size, INT);

    for (size_t h = 0; h < picture.get_size()[0]; h++) {
      for (size_t w = 0; w < picture.get_size()[1]; w++) {
        picture.operator()<int>({h, w}) = rand() % 255;
      }
    }

    return picture;
}