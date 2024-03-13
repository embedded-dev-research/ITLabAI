#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layers/Shape.hpp"

enum Type {
  kInt = 0,
  kDouble,
  kUnknown,
};

template <typename T>
std::vector<uint8_t>* to_byte(std::vector<T>& v) {
  return reinterpret_cast<std::vector<uint8_t>*>(&v);
}

template <typename T>
const std::vector<uint8_t>* to_byte(const std::vector<T>& v) {
  return reinterpret_cast<const std::vector<uint8_t>*>(&v);
}

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

  std::vector<uint8_t> Tensor::SetRightTypeValues() {
    if (type_ == kInt) {
      return std::vector<uint8_t>(shape_.count() * sizeof(int), 0);
    }
    if (type_ == kDouble) {
      return std::vector<uint8_t>(shape_.count() * sizeof(double), 0);
    }
    return std::vector<uint8_t>();
  }

  template <typename T>
  std::vector<T>* as();

 public:
  Tensor(const std::vector<uint8_t>& a, const Shape& s, Type type) {
    type_ = type;
    shape_ = s;
    values_ = SetRightTypeValues();

    if (a.size() != values_.size() || type == kUnknown) {
      throw std::invalid_argument("Wrong_Length");
    }
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

  Tensor(Tensor&& t) = default;
  
  Tensor& operator=(Tensor&& t) = default;

  std::vector<uint8_t> get_values() const { return values_; }
  Shape get_shape() const { return shape_; }
  Type get_type() const noexcept { return type_; }

  template <typename T>
  typename std::vector<T>::const_iterator begin();

  template <typename T>
  typename std::vector<T>::const_iterator end();

  template <typename T>
  T& get(const std::vector<size_t>& coords);  // write

  template <typename T>
  T get(const std::vector<size_t>& coords) const;  // read

  friend std::ostream& operator<<(std::ostream& out, Tensor& t);
};

template <typename T>
std::vector<T>* Tensor::as() {
  if (GetTypeEnum<T>() != type_) {
    throw std::invalid_argument("INVALID_TYPE");
  }
  return reinterpret_cast<std::vector<T>*>(&values_);
}

template <typename T>
typename std::vector<T>::const_iterator begin() {
  return this->as<T>().begin();
}

template <typename T>
typename std::vector<T>::const_iterator end() {
  return this->as<T>().end();
}

std::ostream& operator<<(std::ostream& out, Tensor& t) {
  for (size_t i = 0; i < t.get_shape().count(); i++) {
    out.width(5);
    if (t.get_type() == kInt) out << (*t.as<int>())[i] << " ";
    if (t.get_type() == kDouble) out << (*t.as<double>())[i] << " ";
    if ((i + 1) % t.get_shape()[1] == 0) out << std::endl;
  }

  return out;
}

template <typename T>
T& Tensor::get(const std::vector<size_t>& coords) {
  size_t s = shape_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();

  if ((*res_vector).size() == 0) {
    throw std::invalid_argument("Empty tensor\n");
  }

  return (*res_vector)[s];
}  // write

template <typename T>
T Tensor::get(const std::vector<size_t>& coords) const {
  size_t s = shape_.get_index(coords);
  const std::vector<T>* res_vector = this->as<T>();
  
  if ((*res_vector).size() == 0) {
    throw std::invalid_argument("Empty tensor\n");
  }

  return (*res_vector)[s];
}  // read

template <typename T>
Tensor make_tensor(const std::vector<T>& v) {
  return Tensor(*to_byte<T>(v), {v.size()}, GetTypeEnum<T>());
}

template <typename T>
Tensor make_tensor(const std::vector<T>& v, const Shape& s) {
  return Tensor(*to_byte<T>(v), s, GetTypeEnum<T>());
}