#pragma once
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

const size_t kDepth1 = 128;
const size_t kDepth2 = 5;

class FCLayer : public Layer {
 private:
  Tensor weights_;
  Tensor bias_;
  ImplType implType_;

 public:
  FCLayer() = default;
  FCLayer(Tensor weights, const Tensor& bias, ImplType implType = kDefault)
      : weights_(std::move(weights)), bias_(bias), implType_(implType) {}
  static std::string get_name() { return "Fully-connected layer"; }
  void run(const Tensor& input, Tensor& output) override;
};

template <typename ValueType>
std::vector<ValueType> mat_vec_mul(const std::vector<ValueType>& mat,
                                   const Shape& mat_shape,
                                   const std::vector<ValueType>& vec) {
  if (mat_shape.dims() != 2) {
    throw std::invalid_argument("Not a matrix in argument");
  }
  if (vec.size() < mat_shape[1]) {
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
inline ValueType get_from(size_t i, size_t j, const std::vector<ValueType>& mat,
                          const Shape& mat_shape) {
  if (i < mat_shape[0] && j < mat_shape[1]) {
    return mat[i * mat_shape[1] + j];
  }
  return ValueType(0);
}

template <typename ValueType>
void m_mult(const std::vector<ValueType>& mat,
            const std::vector<ValueType>& vec, const Shape& mat_shape,
            std::vector<ValueType>& res, size_t ind_x, size_t ind_y,
            size_t size, size_t depth) {
  if (depth > kDepth2 || size < kDepth1) {
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        if (ind_x + j < vec.size()) {
          res[ind_y + i] +=
              get_from(ind_y + i, ind_x + j, mat, mat_shape) * vec[ind_x + j];
        }
      }
    }
  } else {
    std::vector<size_t> tmp_x({0, size / 2, 0, size / 2});
    std::vector<size_t> tmp_y({0, 0, size / 2, size / 2});
    for (size_t i = 0; i < 4; i++) {
      m_mult<ValueType>(mat, vec, mat_shape, res, ind_x + tmp_x[i],
                        ind_y + tmp_y[i], size / 2, depth + 1);
    }
  }
}

template <typename ValueType>
void m_mult_tbb(const std::vector<ValueType>& mat,
                const std::vector<ValueType>& vec, const Shape& mat_shape,
                std::vector<ValueType>& res, size_t ind_x, size_t ind_y,
                size_t size, size_t depth) {
  if (depth > kDepth2 || size < kDepth1) {
    for (size_t i = 0; i < size; i++) {
      for (size_t j = 0; j < size; j++) {
        if (ind_x + j < vec.size()) {
          res[ind_y + i] +=
              get_from(ind_y + i, ind_x + j, mat, mat_shape) * vec[ind_x + j];
        }
      }
    }
  } else {
    size_t size_2 = size / 2;
    std::vector<size_t> tmp_x({0, size_2, 0, size_2});
    std::vector<size_t> tmp_y({0, 0, size_2, size_2});
    oneapi::tbb::task_group g;
    g.run([&]() {
      m_mult_tbb<ValueType>(mat, vec, mat_shape, res, ind_x + tmp_x[0],
                            ind_y + tmp_y[0], size_2, depth + 1);
    });
    g.run([&]() {
      m_mult_tbb<ValueType>(mat, vec, mat_shape, res, ind_x + tmp_x[2],
                            ind_y + tmp_y[2], size_2, depth + 1);
    });
    g.wait();
    g.run([&]() {
      m_mult_tbb<ValueType>(mat, vec, mat_shape, res, ind_x + tmp_x[1],
                            ind_y + tmp_y[1], size_2, depth + 1);
    });
    g.run([&]() {
      m_mult_tbb<ValueType>(mat, vec, mat_shape, res, ind_x + tmp_x[3],
                            ind_y + tmp_y[3], size_2, depth + 1);
    });
    g.wait();
  }
}

template <typename ValueType>
std::vector<ValueType> mat_vec_mul_upd(const std::vector<ValueType>& mat,
                                       const Shape& mat_shape,
                                       const std::vector<ValueType>& vec) {
  if (mat_shape.dims() != 2) {
    throw std::invalid_argument("Not a matrix in argument");
  }
  if (vec.size() < mat_shape[1]) {
    throw std::invalid_argument("Invalid vector size");
  }
  size_t near_pow2 = 1;
  while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
    near_pow2 = near_pow2 << 1;
  }
  std::vector<ValueType> res(near_pow2);
  m_mult<ValueType>(mat, vec, mat_shape, res, 0, 0, near_pow2, 1);
  res.resize(mat_shape[0]);
  return res;
}

template <typename ValueType>
std::vector<ValueType> mat_vec_mul_upd_tbb(const std::vector<ValueType>& mat,
                                           const Shape& mat_shape,
                                           const std::vector<ValueType>& vec) {
  if (mat_shape.dims() != 2) {
    throw std::invalid_argument("Not a matrix in argument");
  }
  if (vec.size() < mat_shape[1]) {
    throw std::invalid_argument("Invalid vector size");
  }
  size_t near_pow2 = 1;
  while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
    near_pow2 = near_pow2 << 1;
  }
  std::vector<ValueType> res(near_pow2);
  m_mult_tbb<ValueType>(mat, vec, mat_shape, res, 0, 0, near_pow2, 1);
  res.resize(mat_shape[0]);
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

 protected:
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
  // make weights isize x osize, filling empty with 0s
  weights_.resize(input_weights_shape.count(), ValueType(0));
  //
}

template <typename ValueType>
std::vector<ValueType> FCLayerImpl<ValueType>::run(
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

template <typename ValueType>
class FCLayerImplTBB : public FCLayerImpl<ValueType> {
 public:
  FCLayerImplTBB(const std::vector<ValueType>& input_weights,
                 const Shape& input_weights_shape,
                 const std::vector<ValueType>& input_bias)
      : FCLayerImpl<ValueType>(input_weights, input_weights_shape, input_bias) {
  }
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;
};

template <typename ValueType>
std::vector<ValueType> FCLayerImplTBB<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_[0]) {
    throw std::invalid_argument("Input size doesn't fit FCLayer");
  }
  Shape cur_w_shape({this->outputShape_[0], this->inputShape_[0]});
  std::vector<ValueType> output_values =
      mat_vec_mul_upd_tbb(this->weights_, cur_w_shape, input);
  std::transform(output_values.begin(), output_values.end(),
                 this->bias_.begin(), output_values.begin(),
                 std::plus<ValueType>());
  return output_values;
}

}  // namespace itlab_2023
