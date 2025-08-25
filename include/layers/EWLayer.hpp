#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <utility>

#include "layers/Layer.hpp"

namespace it_lab_ai {

template <typename T>
T relu(const T& value) {
  if (value > T(0)) {
    return value;
  }
  return T(0);
}

class EWLayer : public Layer {
 public:
  EWLayer() = default;
  EWLayer(std::string function, float alpha = 0.0F, float beta = 0.0F)
      : func_(std::move(function)), alpha_(alpha), beta_(beta) {}

  static std::string get_name() { return "Element-wise layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif
 private:
  std::string func_;
  float alpha_;
  float beta_;
};

template <typename ValueType>
class EWLayerImpl : public LayerImpl<ValueType> {
 public:
  EWLayerImpl() = delete;
  EWLayerImpl(const Shape& shape, std::string function, float alpha = 0.0F,
              float beta = 0.0F);
  EWLayerImpl(const EWLayerImpl& c) = default;
  EWLayerImpl& operator=(const EWLayerImpl& c) = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 private:
  std::string func_;
  float alpha_;
  float beta_;
};

template <typename ValueType>
EWLayerImpl<ValueType>::EWLayerImpl(const Shape& shape, std::string function,
                                    float alpha, float beta)
    : LayerImpl<ValueType>(shape, shape),
      func_(std::move(function)),
      alpha_(alpha),
      beta_(beta) {}

template <typename ValueType>
std::vector<ValueType> EWLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  std::vector<ValueType> res(this->outputShape_.count());
  if (func_ == "relu") {
    std::transform(input.begin(), input.end(), res.begin(), relu<ValueType>);
  } else if (func_ == "tanh") {
    auto tanh = [&](const ValueType& value) -> ValueType {
      return static_cast<ValueType>(std::tanh(value));
    };
    std::transform(input.begin(), input.end(), res.begin(), tanh);
  } else if (func_ == "sin") {
    auto sin = [&](const ValueType& value) -> ValueType {
      return static_cast<ValueType>(std::sin(value));
    };
    std::transform(input.begin(), input.end(), res.begin(), sin);
  } else if (func_ == "minus") {
    auto minus = [&](const ValueType& value) -> ValueType { return -value; };
    std::transform(input.begin(), input.end(), res.begin(), minus);
  } else if (func_ == "linear") {
    auto linear = [&](const ValueType& value) -> ValueType {
      return value * static_cast<ValueType>(alpha_) +
             static_cast<ValueType>(beta_);
    };
    std::transform(input.begin(), input.end(), res.begin(), linear);
  } else if (func_ == "sigmoid") {
    auto sigmoid = [](ValueType x) -> ValueType {
      if constexpr (std::is_integral_v<ValueType>) {
        auto x_float = static_cast<float>(x);
        float result = 1.0F / (1.0F + std::exp(-x_float));
        return static_cast<ValueType>(std::round(result));
      } else {
        if (x >= ValueType(0)) {
          ValueType z = std::exp(-x);
          return ValueType(1) / (ValueType(1) + z);
        }
        ValueType z = std::exp(x);
        return z / (ValueType(1) + z);
      }
    };
    std::transform(input.cbegin(), input.cend(), res.begin(), sigmoid);
  } else {
    throw std::invalid_argument("No such function for EWLayer");
  }

  return res;
}

}  // namespace it_lab_ai
