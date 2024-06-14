#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

const size_t kDepth = 64;
void split_into_blocks(std::vector<size_t>& tmp, size_t near_pow2_2);

class FCLayer : public Layer {
 private:
  Tensor weights_;
  Tensor bias_;
  ImplType implType_;

 public:
  FCLayer() = default;
  FCLayer(const Tensor& weights, const Tensor& bias,
          ImplType implType = kDefault)
      : weights_(weights), bias_(bias), implType_(implType) {}
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
std::vector<ValueType> m_plus(const std::vector<ValueType>& mat,
                                     const Shape& mat_shape, size_t ind1,
                                     size_t ind2, size_t size) {
  std::vector<ValueType> res(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      res[i * size + j] = get_from(i, j + ind1, mat, mat_shape) +
                          get_from(i, j + ind2, mat, mat_shape);
    }
  }
  return res;
}

template <typename ValueType>
std::vector<ValueType> m_minus(const std::vector<ValueType>& mat,
                                      const Shape& mat_shape, size_t ind1,
                                      size_t ind2, size_t size) {
  std::vector<ValueType> res(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      res[i * size + j] = get_from(i, j + ind1, mat, mat_shape) -
                          get_from(i, j + ind2, mat, mat_shape);
    }
  }
  return res;
}

template <typename ValueType>
std::vector<ValueType> m_copy(const std::vector<ValueType>& mat,
    const Shape& mat_shape, size_t ind1, size_t size) {
  std::vector<ValueType> res(size * size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      res[i * size + j] = get_from(i, j + ind1, mat, mat_shape);
    }
  }
  return res;
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
  Shape res_shape(1);
  res_shape[0] = mat_shape[0];
  std::vector<ValueType> res;
  if (mat_shape[0] <= kDepth && mat_shape[1] <= kDepth) {
    return mat_vec_mul(mat, mat_shape, vec);
  }
  size_t near_pow2 = 1;
  std::vector<size_t> tmp(4);
  while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
      near_pow2 = near_pow2 << 1;
  }
  size_t near_pow2_2 = near_pow2 / 2;
  split_into_blocks(tmp, near_pow2_2);
  Shape cur_shape({near_pow2_2, near_pow2_2});
  std::vector<ValueType> vec_sec_half(vec.begin() + near_pow2_2, vec.end());
  vec_sec_half.resize(near_pow2_2, ValueType(0));
  std::vector<ValueType> vec2_minus_vec1(vec_sec_half.size());
  std::transform(vec_sec_half.begin(), vec_sec_half.end(), vec.begin(),
                 vec2_minus_vec1.begin(), std::minus<ValueType>());
  std::vector<ValueType> d = mat_vec_mul_upd<ValueType>(
      m_plus(mat, mat_shape, tmp[0], tmp[3], near_pow2_2), cur_shape,
      vec);
  std::vector<ValueType> d1 = mat_vec_mul_upd<ValueType>(
      m_minus(mat, mat_shape, tmp[1], tmp[3], near_pow2_2), cur_shape, vec_sec_half);
  std::vector<ValueType> d2 = mat_vec_mul_upd<ValueType>(
      m_minus(mat, mat_shape, tmp[2], tmp[0], near_pow2_2), cur_shape, vec);
  std::vector<ValueType> h2 = mat_vec_mul_upd<ValueType>(
      m_plus(mat, mat_shape, tmp[2], tmp[3], near_pow2_2), cur_shape, vec);
  std::vector<ValueType> v1 = mat_vec_mul_upd<ValueType>(
      m_copy(mat, mat_shape, tmp[3], near_pow2_2), cur_shape, vec2_minus_vec1);
  std::vector<ValueType> r1(near_pow2_2);
  std::vector<ValueType> r2(near_pow2_2);
  std::transform(d1.begin(), d1.end(), v1.begin(), d1.begin(),
                 std::plus<ValueType>());
  std::transform(d1.begin(), d1.end(), d.begin(), d1.begin(),
                 std::plus<ValueType>());
  std::transform(v1.begin(), v1.end(), h2.begin(), v1.begin(),
                 std::plus<ValueType>());
  res = d1;
  for (size_t i = 0; i < res_shape[0] - d1.size(); i++) {
      res.push_back(v1[i]);
  }
  return res;
}

template <typename ValueType>
std::vector<ValueType> m_plus_tbb(const std::vector<ValueType>& mat,
                                  const Shape& mat_shape, size_t ind1,
                                  size_t ind2, size_t size) {
  std::vector<ValueType> res(size * size);
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(0, size, 0, size),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (size_t i = r.rows().begin(); i < r.rows().end(); i++) {
          for (size_t j = r.cols().begin(); j < r.cols().end(); j++) {
            res[i * size + j] = get_from(i, j + ind1, mat, mat_shape) +
                                get_from(i, j + ind2, mat, mat_shape);
          }
        }
      });
  return res;
}

template <typename ValueType>
std::vector<ValueType> m_minus_tbb(const std::vector<ValueType>& mat,
                                   const Shape& mat_shape, size_t ind1,
                                   size_t ind2, size_t size) {
  std::vector<ValueType> res(size * size);
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(0, size, 0, size),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (size_t i = r.rows().begin(); i < r.rows().end(); i++) {
          for (size_t j = r.cols().begin(); j < r.cols().end(); j++) {
            res[i * size + j] = get_from(i, j + ind1, mat, mat_shape) -
                                get_from(i, j + ind2, mat, mat_shape);
          }
        }
      });
  return res;
}

template <typename ValueType>
std::vector<ValueType> m_copy_tbb(const std::vector<ValueType>& mat,
                              const Shape& mat_shape, size_t ind1,
                              size_t size) {
  std::vector<ValueType> res(size * size);
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(0, size, 0, size),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (size_t i = r.rows().begin(); i < r.rows().end(); i++) {
          for (size_t j = r.cols().begin(); j < r.cols().end(); j++) {
            res[i * size + j] = get_from(i, j + ind1, mat, mat_shape);
          }
        }
      });
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
  Shape res_shape(1);
  res_shape[0] = mat_shape[0];
  std::vector<ValueType> res;
  if (mat_shape[0] <= kDepth && mat_shape[1] <= kDepth) {
    return mat_vec_mul(mat, mat_shape, vec);
  }
  size_t near_pow2 = 1;
  std::vector<size_t> tmp(4);
  while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
    near_pow2 = near_pow2 << 1;
  }
  size_t near_pow2_2 = near_pow2 / 2;
  // split_into_blocks(tmp, near_pow2_2);
  tmp[0] = 0;
  tmp[1] = near_pow2_2;
  tmp[2] = 2 * near_pow2_2 * near_pow2_2;
  tmp[3] = (near_pow2_2) * (2 * near_pow2_2 + 1);
  Shape cur_shape({near_pow2_2, near_pow2_2});
  std::vector<ValueType> vec_sec_half(vec.begin() + near_pow2_2, vec.end());
  vec_sec_half.resize(near_pow2_2, ValueType(0));
  std::vector<ValueType> vec2_minus_vec1(vec_sec_half.size());
  std::vector<ValueType> d;
  std::vector<ValueType> d1;
  std::vector<ValueType> d2;
  std::vector<ValueType> h2;
  std::vector<ValueType> v1;
  oneapi::tbb::task_group g;
  g.run([&]() {
    std::transform(vec_sec_half.begin(), vec_sec_half.end(), vec.begin(),
                   vec2_minus_vec1.begin(), std::minus<ValueType>());
  });
  g.run([&]() {
    d = mat_vec_mul<ValueType>(
        m_plus_tbb(mat, mat_shape, tmp[0], tmp[3], near_pow2_2), cur_shape,
        vec);
  });
  g.run([&]() {
    d1 = mat_vec_mul<ValueType>(
        m_minus_tbb(mat, mat_shape, tmp[1], tmp[3], near_pow2_2), cur_shape,
        vec_sec_half);
  });
  g.run([&]() {
    d2 = mat_vec_mul<ValueType>(
        m_minus_tbb(mat, mat_shape, tmp[2], tmp[0], near_pow2_2), cur_shape,
        vec);
  });
  g.run([&]() {
    h2 = mat_vec_mul<ValueType>(
        m_plus_tbb(mat, mat_shape, tmp[2], tmp[3], near_pow2_2), cur_shape,
        vec);
  });
  g.wait();
  v1 = mat_vec_mul<ValueType>(
      m_copy_tbb(mat, mat_shape, tmp[3], near_pow2_2), cur_shape,
      vec2_minus_vec1);
  std::transform(d1.begin(), d1.end(), v1.begin(), d1.begin(),
                 std::plus<ValueType>());
  std::transform(d1.begin(), d1.end(), d.begin(), d1.begin(),
                 std::plus<ValueType>());
  std::transform(v1.begin(), v1.end(), h2.begin(), v1.begin(),
                 std::plus<ValueType>());
  res = d1;
  for (size_t i = 0; i < res_shape[0] - d1.size(); i++) {
    res.push_back(v1[i]);
  }
  return res;
}

template <typename ValueType>
std::vector<ValueType> mat_vec_mul_tbb(const std::vector<ValueType>& mat,
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
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(0, mat_shape[0], 0, mat_shape[1]),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (size_t i = r.rows().begin(); i < r.rows().end(); i++) {
          elem = ValueType(0);
          for (size_t j = r.cols().begin(); j < r.cols().end(); j++) {
            // due to 1d indexing
            elem += mat[i * mat_shape[1] + j] * vec[j];
          }
          res[i] = elem;
        }
      });
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
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

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
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;
};

template <typename ValueType>
std::vector<ValueType> FCLayerImplTBB<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_[0]) {
    throw std::invalid_argument("Input size doesn't fit FCLayer");
  }
  Shape cur_w_shape({this->outputShape_[0], this->inputShape_[0]});
  std::vector<ValueType> output_values =
      mat_vec_mul_tbb(this->weights_, cur_w_shape, input);
  std::transform(output_values.begin(), output_values.end(),
                 this->bias_.begin(), output_values.begin(),
                 std::plus<ValueType>());
  return output_values;
}

}  // namespace itlab_2023
