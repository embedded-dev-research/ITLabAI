#pragma once

#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>

#include "Layer.hpp"

enum Type {
  kInt = 0,
  kDouble,
};

template <typename T>
Type GetTypeEnum() {
  if (std::is_same<T, int>::value)
    return kInt;
  if (std::is_same<T, int>::value)
    return kDouble;
  throw std::invalid_argument("invalid type\n");
}

class Tensor {
private:
  Shape shape_;
  std::vector<uint8_t> values_;
  Type type_;

  std::vector<uint8_t> SetRightTypeValues();

  template <typename T>
  std::vector<T>* as();
public:
  Tensor(const size_t dims_count, Type type) : shape_(dims_count) {
    type_ = type;
    SetRightTypeValues();
  };
  Tensor(const std::vector<size_t>& dims, Type type) : shape_(dims) {
    type_ = type;
    SetRightTypeValues();
  }
  Tensor(const Shape& sh, Type type) : shape_(sh) {
    type_ = type;
    SetRightTypeValues();
  }
  Tensor(const Tensor& t) = default;

  Shape get_size() const { return shape_; }

  template <typename T>
  T& operator()(const std::vector<size_t>& coords); // write
  template <typename T>
  T operator()(const std::vector<size_t>& coords) const; // read

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

