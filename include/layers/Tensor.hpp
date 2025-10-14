#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "layers/Shape.hpp"

namespace it_lab_ai {

enum class Type : uint8_t { kUnknown, kInt, kInt64, kFloat };

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
  if constexpr (std::is_same_v<T, int>) {
    return Type::kInt;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return Type::kInt64;
  } else if constexpr (std::is_same_v<T, float>) {
    return Type::kFloat;
  } else {
    return Type::kUnknown;
  }
}
class Tensor {
 private:
  Shape shape_;
  std::vector<uint8_t> values_;
  std::vector<float> bias_;
  Type type_ = Type::kUnknown;

  std::vector<uint8_t> SetRightTypeValues() {
    if (type_ == Type::kInt) {
      return std::vector<uint8_t>(shape_.count() * sizeof(int), 0);
    }
    if (type_ == Type::kFloat) {
      return std::vector<uint8_t>(shape_.count() * sizeof(float), 0);
    }
    return std::vector<uint8_t>();
  }

 public:
  Tensor() = default;

  Tensor(const std::vector<uint8_t>& a, const Shape& sh, Type type)
      : shape_(sh), type_(type) {
    if (type == Type::kUnknown) {
      throw std::invalid_argument("Unknown data type");
    }

    values_ = SetRightTypeValues();

    if (a.size() != values_.size()) {
      throw std::invalid_argument("Incorrect vector size given to Tensor");
    }

    values_ = a;
  }

  Tensor(const Shape& sh, Type type) : shape_(sh), type_(type) {
    if (type == Type::kUnknown) {
      throw std::invalid_argument("Unknown data type");
    }

    values_ = SetRightTypeValues();
  }

  Tensor(const std::vector<uint8_t>& a, const Shape& sh,
         const std::vector<float>& bias)
      : shape_(sh), values_(a), bias_(bias), type_(Type::kFloat) {
    values_ = SetRightTypeValues();
    if (a.size() != values_.size()) {
      throw std::invalid_argument("Incorrect vector size given to Tensor");
    }
    values_ = a;
  }

  Tensor(const Tensor& t) = default;
  Tensor(Tensor&& t) = default;
  Tensor& operator=(Tensor&& t) = default;
  Tensor& operator=(const Tensor& t) = default;

  Shape get_shape() const { return shape_; }
  Type get_type() const noexcept { return type_; }

  void set_bias(const std::vector<float>& bias) {
    if (bias.size() != shape_[shape_.dims() - 1]) {
      throw std::invalid_argument(
          "Bias size does not match the last dimension of the shape");
    }
    bias_ = bias;
  }

  const std::vector<float>& get_bias() const { return bias_; }
  const std::vector<uint8_t>& get_values() const { return values_; }

  bool empty() const { return values_.empty(); }
  auto begin() { return values_.begin(); }

  auto end() { return values_.end(); }

  auto begin() const { return values_.begin(); }

  auto end() const { return values_.end(); }

  template <typename T>
  typename std::vector<T>::const_iterator begin() const {
    return this->as<T>().begin();
  }

  template <typename T>
  typename std::vector<T>::const_iterator end() const {
    return this->as<T>().end();
  }

  template <typename T>
  void set(const std::vector<size_t>& coords, const T& elem);

  template <typename T>
  T get(const std::vector<size_t>& coords) const;

  template <typename T>
  std::vector<T>* as();

  template <typename T>
  const std::vector<T>* as() const;

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

template <typename T>
void Tensor::set(const std::vector<size_t>& coords, const T& elem) {
  size_t s = shape_.get_index(coords);
  std::vector<T>* res_vector = this->as<T>();
  if ((*res_vector).size() == 0) {
    throw std::invalid_argument("Empty tensor");
  }
  (*res_vector)[s] = elem;
}

template <typename T>
T Tensor::get(const std::vector<size_t>& coords) const {
  size_t s = shape_.get_index(coords);
  const std::vector<T>* res_vector = this->as<T>();
  if ((*res_vector).size() == 0) {
    throw std::invalid_argument("Empty tensor");
  }
  return (*res_vector)[s];
}

template <typename T>
std::vector<T>* Tensor::as() {
  if (GetTypeEnum<T>() != type_) {
    throw std::invalid_argument("Template type doesn't fit this Tensor");
  }
  return reinterpret_cast<std::vector<T>*>(&values_);
}

template <typename T>
const std::vector<T>* Tensor::as() const {
  if (GetTypeEnum<T>() != type_) {
    throw std::invalid_argument("Template type doesn't fit this Tensor");
  }
  return reinterpret_cast<const std::vector<T>*>(&values_);
}

template <typename T>
Tensor make_tensor(const std::vector<T>& values) {
  Type type = GetTypeEnum<T>();
  if (type == Type::kUnknown) {
    throw std::invalid_argument("Unsupported tensor type");
  }

  Shape shape({values.size()});
  std::vector<uint8_t> byte_values(
      reinterpret_cast<const uint8_t*>(values.data()),
      reinterpret_cast<const uint8_t*>(values.data() + values.size()));

  return Tensor(byte_values, shape, type);
}

template <typename T>
Tensor make_tensor(const std::vector<T>& values, const Shape& shape) {
  std::vector<uint8_t> byte_values(
      reinterpret_cast<const uint8_t*>(values.data()),
      reinterpret_cast<const uint8_t*>(values.data() + values.size()));
  return Tensor(byte_values, shape, GetTypeEnum<T>());
}

template <typename T>
Tensor make_tensor(const std::vector<T>& values, const Shape& shape,
                   const std::vector<float>& bias) {
  std::vector<uint8_t> byte_values(
      reinterpret_cast<const uint8_t*>(values.data()),
      reinterpret_cast<const uint8_t*>(values.data() + values.size()));
  return Tensor(byte_values, shape, bias);
}
std::ostream& operator<<(std::ostream& out, const Tensor& t);

}  // namespace it_lab_ai
