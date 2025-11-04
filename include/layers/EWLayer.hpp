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
  EWLayer() : Layer(kElementWise), func_("none"), alpha_(0.0F), beta_(0.0F) {}
  EWLayer(std::string function, float alpha = 0.0F, float beta = 0.0F)
      : Layer(kElementWise),
        func_(std::move(function)),
        alpha_(alpha),
        beta_(beta) {}

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
              float beta = 0.0F, int type_parall = 0);
  EWLayerImpl(const EWLayerImpl& c) = default;
  EWLayerImpl& operator=(const EWLayerImpl& c) = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 private:
  std::string func_;
  float alpha_;
  float beta_;
  int type_parall_;
};

template <typename ValueType>
EWLayerImpl<ValueType>::EWLayerImpl(const Shape& shape, std::string function,
                                    float alpha, float beta, int type_parall)
    : LayerImpl<ValueType>(shape, shape),
      func_(std::move(function)),
      alpha_(alpha),
      beta_(beta),
      type_parall_(type_parall) {}

template <typename ValueType>
std::vector<ValueType> EWLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  std::vector<ValueType> res(this->outputShape_.count());
  int available_threads = -1;
  if (type_parall_ == 0) available_threads = 1;
  if (type_parall_ == 1)
    available_threads = std::thread::hardware_concurrency();
  if (type_parall_ == 2)
    available_threads = oneapi::tbb::info::default_concurrency();
  if (type_parall_ == 3) available_threads = omp_get_max_threads();

  if (func_ == "relu") {
    parallel_for(
        input.size(),
        [&](int i) {
          res[i] = input[i] > ValueType(0) ? input[i] : ValueType(0);
        },
        type_parall_);
  } else if (func_ == "tanh") {
    parallel_for(
        input.size(),
        [&](int i) { res[i] = static_cast<ValueType>(std::tanh(input[i])); },
        type_parall_);
  } else if (func_ == "sin") {
    parallel_for(
        input.size(),
        [&](int i) { res[i] = static_cast<ValueType>(std::sin(input[i])); },
        type_parall_);
  } else if (func_ == "minus") {
    parallel_for(
        input.size(), [&](int i) { res[i] = -input[i]; }, type_parall_);
  } else if (func_ == "linear") {
    parallel_for(
        input.size(),
        [&](int i) {
          res[i] = input[i] * static_cast<ValueType>(alpha_) +
                   static_cast<ValueType>(beta_);
        },
        type_parall_);
  } else if (func_ == "sigmoid") {
    if constexpr (std::is_integral_v<ValueType>) {
      parallel_for(
          input.size(),
          [&](int i) {
            auto x_float = static_cast<float>(input[i]);
            float result = 1.0F / (1.0F + std::exp(-x_float));
            res[i] = static_cast<ValueType>(std::round(result));
          },
          type_parall_);
    } else {
      parallel_for(
          input.size(),
          [&](int i) {
            ValueType x = input[i];
            if (x >= ValueType(0)) {
              ValueType z = std::exp(-x);
              res[i] = ValueType(1) / (ValueType(1) + z);
            } else {
              ValueType z = std::exp(x);
              res[i] = z / (ValueType(1) + z);
            }
          },
          type_parall_);
    }
  } else {
    throw std::invalid_argument("No such function for EWLayer");
  }

  return res;
}

}  // namespace it_lab_ai
