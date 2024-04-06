#pragma once
#include <algorithm>
#include <cmath>
#include <string>

#include "layers/Layer.hpp"

class EWLayer : public Layer {
 public:
  EWLayer() = default;
  EWLayer(const std::string& function, float alpha = 0.0F, float beta = 0.0F)
      : func_(function), alpha_(alpha), beta_(beta) {}
  static std::string get_name() { return "Element-wise layer"; }
  void run(const Tensor& input, Tensor& output);
 private:
  std::string func_;
  float alpha_;
  float beta_;
};

template <typename T>
T minus(const T& elem) {
  return -elem;
}

template <typename T>
T mysin(const T& elem) {
  return static_cast<T>(std::sin(elem));
}

template <typename T>
T mytanh(const T& elem) {
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
  EWLayerImpl(const Shape& shape, const std::string& function,
              float alpha = 0.0F, float beta = 0.0F);
  EWLayerImpl(const EWLayerImpl& c) = default;
  EWLayerImpl& operator=(const EWLayerImpl& c) = default;
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;
 private:
  std::string func_;
  float alpha_;
  float beta_;
};

template <typename ValueType>
EWLayerImpl<ValueType>::EWLayerImpl(const Shape& shape,
                                    const std::string& function, float alpha,
                                    float beta)
    : LayerImpl<ValueType>(shape, shape),
      func_(function),
      alpha_(alpha),
      beta_(beta) {}

template <typename ValueType>
std::vector<ValueType> EWLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  std::vector<ValueType> res(this->outputShape_.count());
  if (func_ == "relu") {
    std::transform(input.begin(), input.end(), res.begin(), relu<ValueType>);
  } else if (func_ == "tanh") {
    std::transform(input.begin(), input.end(), res.begin(), mytanh<ValueType>);
  } else if (func_ == "sin") {
    std::transform(input.begin(), input.end(), res.begin(), mysin<ValueType>);
  } else if (func_ == "minus") {
    std::transform(input.begin(), input.end(), res.begin(), minus<ValueType>);
  } else if (func_ == "linear") {
    auto linear = [&](const ValueType& value) -> ValueType {
      return value * static_cast<ValueType>(alpha_) +
             static_cast<ValueType>(beta_);
    };
    std::transform(input.begin(), input.end(), res.begin(), linear);
  } else {
    throw std::invalid_argument("No such function for EWLayer");
  }
  return res;
}
