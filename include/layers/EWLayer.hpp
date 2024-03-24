#pragma once
#include <algorithm>
#include <cmath>
#include <string>

#include "layers/Layer.hpp"

class EWLayer : public Layer {
 public:
  EWLayer() = default;
  static std::string get_name() { return "Element-wise layer"; }
  void run(const Tensor& input, Tensor& output, const std::string& function);
};

template <typename T>
T minus(const T& elem) {
  return -elem;
}

template <typename T>
T sin(const T& elem) {
  return static_cast<T>(std::sin(elem));
}

template <typename T>
T tanh(const T& elem) {
  return static_cast<T>(std::tanh(elem));
}

template <typename T>
T relu(const T& value) {
  if (value > T(0)) {
    return value;
  }
  return T(0);
}

template <typename ValueType>
class EWLayerImpl : public LayerImpl<ValueType> {
 public:
  EWLayerImpl() = delete;
  EWLayerImpl(const Shape& shape, const std::string& function);
  EWLayerImpl(const EWLayerImpl& c) = default;
  EWLayerImpl& operator=(const EWLayerImpl& c) = default;
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

 private:
  ValueType (*unaryFunc_)(const ValueType&);
};

template <typename ValueType>
EWLayerImpl<ValueType>::EWLayerImpl(const Shape& shape,
                                    const std::string& function)
    : LayerImpl<ValueType>(shape, shape) {
  if (function == "relu") {
    unaryFunc_ = relu<ValueType>;
  } else if (function == "tanh") {
    unaryFunc_ = tanh<ValueType>;
  } else if (function == "sin") {
    unaryFunc_ = sin<ValueType>;
  } else if (function == "minus") {
    unaryFunc_ = minus<ValueType>;
  } else {
    throw std::invalid_argument("No such function for EWLayer");
  }
}

template <typename ValueType>
std::vector<ValueType> EWLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  std::vector<ValueType> res(this->outputShape_.count());
  std::transform(input.begin(), input.end(), res.begin(), unaryFunc_);
  return res;
}
