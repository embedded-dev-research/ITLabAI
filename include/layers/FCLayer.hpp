#pragma once
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

class Shape {
 public:
  Shape() = default;
  Shape(size_t dims_count) : dims_(dims_count, 0) {}
  Shape(const std::vector<size_t>& dims) : dims_(dims) {}
  Shape(const std::initializer_list<size_t>& l) : dims_(l) {}
  Shape(const Shape& c) = default;
  Shape& operator=(const Shape& c) = default;
  size_t operator[](size_t i) const noexcept { return dims_[i]; }
  size_t& operator[](size_t i) noexcept { return dims_[i]; }
  size_t at(size_t i) const {
    if (i >= dims_.size()) {
      throw std::out_of_range("Invalid shape index");
    }
    return dims_[i];
  }
  size_t& at(size_t i) {
    if (i >= dims_.size()) {
      throw std::out_of_range("Invalid shape index");
    }
    return dims_[i];
  }
  void resize(const std::vector<size_t>& new_size) { dims_ = new_size; }
  size_t count() const {
    return std::accumulate(dims_.begin(), dims_.end(), size_t(1),
                           std::multiplies<>());
  }
  size_t dims() const noexcept { return dims_.size(); }
  size_t get_index(const std::vector<size_t>& coords) const;

 private:
  std::vector<size_t> dims_;
};

template <typename ValueType>
std::vector<ValueType> mat_vec_mul(const std::vector<ValueType>& mat,
                                   const Shape& mat_shape,
                                   const std::vector<ValueType>& vec) {
  if (mat_shape.dims() != 2) {
    throw std::invalid_argument("Not a matrix in argument");
  }
  if (vec.size() != mat_shape[1]) {
    throw std::invalid_argument("Invalid vector size");
  }
  Shape res_shape(1);
  res_shape[0] = mat_shape[0];
  std::vector<ValueType> res(res_shape[0]);
  ValueType elem;
  for (size_t i = 0; i < mat_shape[0]; i++) {
    elem = ValueType(0);
    for (size_t j = 0; j < mat_shape[1]; j++) {
      // due to 1d indexing
      elem += mat[i * mat_shape[1] + j] * vec[j];
    }
    res[i] = elem;
  }
  return res;
}

template <typename ValueType>
class Layer {
 public:
  Layer() = default;
  Layer(const Shape& inputShape, const Shape& outputShape)
      : inputShape_(inputShape), outputShape_(outputShape) {}
  Layer(const Layer& c) = default;
  Layer& operator=(const Layer& c) = default;
  virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
  Shape get_input_shape() const { return inputShape_; }
  Shape get_output_shape() const { return outputShape_; }
  // weights width x height
  std::pair<Shape, Shape> get_dims() const {
    return std::pair<Shape, Shape>(outputShape_, inputShape_);
  }

 protected:
  Shape inputShape_;
  Shape outputShape_;
};

template <typename ValueType>
class FCLayer : public Layer<ValueType> {
 public:
  FCLayer() = delete;
  FCLayer(const std::vector<ValueType>& input_weights,
          const Shape& input_weights_shape,
          const std::vector<ValueType>& input_bias);
  FCLayer(const FCLayer& c) = default;
  FCLayer& operator=(const FCLayer& sec) = default;
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
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

 private:
  std::vector<ValueType> weights_;
  std::vector<ValueType> bias_;
};

// weights * inputValues + bias = outputValues

// constructor for FCLayer
template <typename ValueType>
FCLayer<ValueType>::FCLayer(const std::vector<ValueType>& input_weights,
                            const Shape& input_weights_shape,
                            const std::vector<ValueType>& input_bias)
    : weights_(input_weights), bias_(input_bias), Layer<ValueType>(1, 1) {
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
  // make weights isize x osize, filling empty with 0s
  weights_.resize(input_weights_shape.count(), ValueType(0));
  //
}

template <typename ValueType>
std::vector<ValueType> FCLayer<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_[0]) {
    throw std::invalid_argument("Input size doesn't fit FCLayer");
  }
  Shape cur_w_shape({this->outputShape_[0], this->inputShape_[0]});
  std::vector<ValueType> output_values =
      mat_vec_mul(weights_, cur_w_shape, input);
  std::transform(output_values.begin(), output_values.end(), bias_.begin(),
                 output_values.begin(), std::plus<ValueType>());
  return output_values;
}
