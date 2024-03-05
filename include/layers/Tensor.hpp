#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Layer.hpp"

enum Type {
  kInt = 0,
  kDouble,
  kUnknown,
};

template <typename T>
Type GetTypeEnum() {
  if (std::is_same<T, int>::value) return kInt;
  if (std::is_same<T, double>::value) return kDouble;
  return kUnknown;
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
    values_ = SetRightTypeValues();
  };
  Tensor(const std::vector<uint8_t>& a, const Shape& s, Type type) {
    type_ = type;
    shape_ = s;
    values_ = SetRightTypeValues();

    if (a.size() != values_.size()) throw std::invalid_argument("Wrong_Length");
    values_ = a;
  }
  Tensor(const std::vector<size_t>& dims, Type type) : shape_(dims) {
    type_ = type;
    values_ = SetRightTypeValues();
  }
  Tensor(const Shape& sh, Type type) : shape_(sh) {
    type_ = type;
    values_ = SetRightTypeValues();
  }
  Tensor(const Tensor& t) = default;

  Shape get_shape() const { return shape_; }
  Type get_type() const noexcept { return type_; }

  template <typename T>
  T& operator()(const std::vector<size_t>& coords);  // write
  template <typename T>
  T operator()(const std::vector<size_t>& coords) const;  // read

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};