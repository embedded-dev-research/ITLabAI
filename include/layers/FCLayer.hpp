#pragma once
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class FCLayer : public Layer {
 private:
  Tensor weights_;
  Tensor bias_;

 public:
  FCLayer() = default;
  FCLayer(Tensor weights, const Tensor& bias)
      : weights_(std::move(weights)), bias_(bias) {}
  static std::string get_name() { return "Fully-connected layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return weights_; }
#endif
};

template <typename ValueType>
std::vector<ValueType> mat_vec_mul(const std::vector<ValueType>& mat,
                                   const Shape& mat_shape,
                                   const std::vector<ValueType>& vec) {
  size_t c = vec.size() / mat_shape[1];
  if (mat_shape.dims() != 2) {
    throw std::invalid_argument("Not a matrix in argument");
  }
  Shape res_shape(1);
  res_shape[0] = mat_shape[0] * c;
  std::vector<ValueType> res(res_shape[0]);
  ValueType elem;
  for (size_t count = 0; count < c; count++) {
    for (size_t i = 0; i < mat_shape[0]; i++) {
      elem = ValueType(0);
      for (size_t j = 0; j < mat_shape[1]; j++) {
        // due to 1d indexing
        elem += mat[i * mat_shape[1] + j] * vec[count * mat_shape[1] + j];
      }
      res[count * mat_shape[0] + i] = elem;
    }
  }
  return res;
}

template <typename ValueType>
class FCLayerImpl : public LayerImpl<ValueType> {
 public:
  FCLayerImpl() = delete;
  FCLayerImpl(const std::vector<ValueType>& input_weights,
              const Shape& input_weights_shape,
              const std::vector<ValueType>& input_bias);
  FCLayerImpl(const FCLayerImpl& c) = default;
  FCLayerImpl& operator=(const FCLayerImpl& sec) = default;
  void set_weight(size_t i, size_t j, const ValueType& value) {
    if (i >= this->outputShape_[0] || j >= this->inputShape_[0]) {
      throw std::out_of_range("Invalid weight index");
    }
    weights_[i * this->inputShape_[0] + j] = value;
  }
  ValueType get_weight(size_t i, size_t j) const {
    if (i >= this->outputShape_[0] || j >= this->inputShape_[0]) {
      throw std::out_of_range("Invalid weight index");
    }
    return weights_[i * this->inputShape_[0] + j];
  }
  void set_bias(size_t i, const ValueType& value) {
    if (i >= this->outputShape_[0]) {
      throw std::out_of_range("Invalid bias index");
    }
    bias_[i] = value;
  }
  ValueType get_bias(size_t i) const {
    if (i >= this->outputShape_[0]) {
      throw std::out_of_range("Invalid bias index");
    }
    return bias_[i];
  }
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 private:
  std::vector<ValueType> weights_;
  std::vector<ValueType> bias_;
};

// weights * inputValues + bias = outputValues

// constructor for FCLayer
template <typename ValueType>
FCLayerImpl<ValueType>::FCLayerImpl(const std::vector<ValueType>& input_weights,
                                    const Shape& input_weights_shape,
                                    const std::vector<ValueType>& input_bias)
    : LayerImpl<ValueType>(1, 1), weights_(input_weights), bias_(input_bias) {
  if (input_weights.empty()) {
    throw std::invalid_argument("Empty weights for FCLayer");
  }
  if (input_weights_shape.dims() != 2 ||
      input_weights_shape[0] != input_bias.size()) {
    throw std::invalid_argument("Invalid weights shape");
  }
  this->inputShape_[0] = input_weights_shape[1];
  this->outputShape_[0] = input_bias.size();
  if (this->inputShape_[0] == 0 || this->outputShape_[0] == 0) {
    throw std::invalid_argument("Invalid weights/bias size for FCLayer");
  }
  weights_.resize(input_weights_shape.count(), ValueType(0));
}

template <typename ValueType>
std::vector<ValueType> FCLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  Shape cur_w_shape({this->outputShape_[0], this->inputShape_[0]});
  std::vector<ValueType> output_values =
      mat_vec_mul(weights_, cur_w_shape, input);
  for (size_t p = 0; p < output_values.size() / bias_.size(); ++p) {
    for (size_t i = 0; i < bias_.size(); ++i) {
      output_values[p * bias_.size() + i] += bias_[i];
    }
  }
  return output_values;
}

}  // namespace it_lab_ai
