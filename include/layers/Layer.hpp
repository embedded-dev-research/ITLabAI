#pragma once
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layers/Shape.hpp"
#include "layers/Tensor.hpp"

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
