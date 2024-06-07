#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

const size_t DEPTH = 64;

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
inline ValueType get_from(size_t i, size_t j, const std::vector<ValueType>& mat,
                   const Shape& mat_shape) {
  if (i < mat_shape[0] && j < mat_shape[1]) {
    return mat[i * mat_shape[1] + j];
  }
  return ValueType(0);
}

template <typename ValueType>
inline std::vector<ValueType> m_plus(const std::vector<ValueType>& mat1,
                                     const std::vector<ValueType>& mat2) {
  std::vector<ValueType> res(mat1.size());
  std::transform(mat1.begin(), mat1.end(), mat2.begin(), res.begin(),
                 std::plus<ValueType>());
  return res;
}

template <typename ValueType>
inline std::vector<ValueType> m_minus(const std::vector<ValueType>& mat1,
                                      const std::vector<ValueType>& mat2) {
  std::vector<ValueType> res(mat1.size());
  std::transform(mat1.begin(), mat1.end(), mat2.begin(), res.begin(),
                 std::minus<ValueType>());
  return res;
}

template <typename ValueType>
void split_into_blocks(const std::vector<ValueType>& mat,
                       const Shape& mat_shape,
                       const std::vector<ValueType>& vec,
                       std::vector<std::vector<ValueType> >& tmp,
                       size_t near_pow2) {
  for (size_t i = 0; i < near_pow2 / 2; i++) {
    for (size_t j = 0; j < near_pow2 / 2; j++) {
      tmp[0].push_back(get_from<ValueType>(i, j, mat, mat_shape));
    }
    for (size_t j = near_pow2 / 2; j < near_pow2; j++) {
      tmp[1].push_back(get_from<ValueType>(i, j, mat, mat_shape));
    }
  }
  for (size_t i = near_pow2 / 2; i < near_pow2; i++) {
    for (size_t j = 0; j < near_pow2 / 2; j++) {
      tmp[2].push_back(get_from<ValueType>(i, j, mat, mat_shape));
    }
    for (size_t j = near_pow2 / 2; j < near_pow2; j++) {
      tmp[3].push_back(get_from<ValueType>(i, j, mat, mat_shape));
    }
  }
  for (size_t i = 0; i < near_pow2 / 2; i++) {
    tmp[4].push_back(get_from<ValueType>(0, i, vec, mat_shape));
  }
  for (size_t i = near_pow2 / 2; i < near_pow2; i++) {
    tmp[5].push_back(get_from<ValueType>(0, i, vec, mat_shape));
  }
}

template <typename ValueType>
std::vector<ValueType> mat_vec_mul_upd(const std::vector<ValueType>& mat,
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
  std::vector<ValueType> res;
  if (mat_shape[0] <= DEPTH && mat_shape[1] <= DEPTH) {
    return mat_vec_mul(mat, mat_shape, vec);
  } else {
    size_t near_pow2 = 1;
    std::vector<std::vector<ValueType> > tmp(6);
    while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
      near_pow2 = near_pow2 << 1;
    }
    split_into_blocks(mat, mat_shape, vec, tmp, near_pow2);
    Shape cur_shape({near_pow2 / 2, near_pow2 / 2});
    std::vector<ValueType> d =
        mat_vec_mul_upd<ValueType>(m_plus(tmp[0], tmp[3]), cur_shape, tmp[4]);
    std::vector<ValueType> d1 =
        mat_vec_mul_upd<ValueType>(m_minus(tmp[1], tmp[3]), cur_shape, tmp[5]);
    std::vector<ValueType> d2 =
        mat_vec_mul_upd<ValueType>(m_minus(tmp[2], tmp[0]), cur_shape, tmp[4]);
    std::vector<ValueType> h2 =
        mat_vec_mul_upd<ValueType>(m_plus(tmp[2], tmp[3]), cur_shape, tmp[4]);
    std::vector<ValueType> v1 =
        mat_vec_mul_upd<ValueType>(tmp[3], cur_shape, m_minus(tmp[5], tmp[4]));
    std::vector<ValueType> r1 = m_plus(m_plus(d1, v1), d);
    std::vector<ValueType> r2 = m_plus(v1, h2);
    res = r1;
    for (size_t i = 0; i < res_shape[0] - r1.size(); i++) {
      res.push_back(r2[i]);
    }
  }
  return res;
}

template <typename ValueType>
void split_into_blocks_tbb(const std::vector<ValueType>& mat,
                           const Shape& mat_shape,
                           const std::vector<ValueType>& vec,
                           std::vector<std::vector<ValueType> >& tmp,
                           size_t near_pow2) {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, near_pow2 / 2),
      [&](oneapi::tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          for (size_t j = 0; j < near_pow2 / 2; j++) {
            tmp[0][i * (near_pow2 / 2) + j] =
                get_from<ValueType>(i, j, mat, mat_shape);
          }
          for (size_t j = near_pow2 / 2; j < near_pow2; j++) {
            tmp[1][i * (near_pow2 / 2) + j - near_pow2 / 2] =
                get_from<ValueType>(i, j, mat, mat_shape);
          }
        }
      });
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(near_pow2 / 2, near_pow2),
      [&](oneapi::tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          for (size_t j = 0; j < near_pow2 / 2; j++) {
            tmp[2][(i - near_pow2 / 2) * (near_pow2 / 2) + j] =
                get_from<ValueType>(i, j, mat, mat_shape);
          }
          for (size_t j = near_pow2 / 2; j < near_pow2; j++) {
            tmp[3][(i - near_pow2 / 2) * (near_pow2 / 2) + j - near_pow2 / 2] =
                get_from<ValueType>(i, j, mat, mat_shape);
          }
        }
      });
  for (size_t i = 0; i < near_pow2 / 2; i++) {
    tmp[4].push_back(get_from<ValueType>(0, i, vec, mat_shape));
  }
  for (size_t i = near_pow2 / 2; i < near_pow2; i++) {
    tmp[5].push_back(get_from<ValueType>(0, i, vec, mat_shape));
  }
}

template <typename ValueType>
std::vector<ValueType> mat_vec_mul_upd_tbb(const std::vector<ValueType>& mat,
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
  std::vector<ValueType> res;
  if (mat_shape[0] <= DEPTH && mat_shape[1] <= DEPTH) {
    return mat_vec_mul(mat, mat_shape, vec);
  } else {
    size_t near_pow2 = 1;
    while (near_pow2 < mat_shape[0] || near_pow2 < mat_shape[1]) {
      near_pow2 = near_pow2 << 1;
    }
    std::vector<std::vector<ValueType> > tmp(
        4, std::vector<ValueType>((near_pow2 / 2) * (near_pow2 / 2)));
    tmp.push_back(std::vector<ValueType>());
    tmp.push_back(std::vector<ValueType>());
    split_into_blocks_tbb(mat, mat_shape, vec, tmp, near_pow2);
    Shape cur_shape({near_pow2 / 2, near_pow2 / 2});
    oneapi::tbb::task_group g;
    std::vector<ValueType> d;
    std::vector<ValueType> d1;
    std::vector<ValueType> d2;
    std::vector<ValueType> h2;
    std::vector<ValueType> v1;
    g.run([&]() {
      d = mat_vec_mul_upd_tbb<ValueType>(m_plus(tmp[0], tmp[3]), cur_shape,
                                         tmp[4]);
    });
    g.run([&]() {
      d1 = mat_vec_mul_upd_tbb<ValueType>(m_minus(tmp[1], tmp[3]), cur_shape,
                                          tmp[5]);
    });
    g.run([&]() {
      d2 = mat_vec_mul_upd_tbb<ValueType>(m_minus(tmp[2], tmp[0]), cur_shape,
                                          tmp[4]);
    });
    g.run([&]() {
      h2 = mat_vec_mul_upd_tbb<ValueType>(m_plus(tmp[2], tmp[3]), cur_shape,
                                          tmp[4]);
    });
    g.run([&]() {
      v1 = mat_vec_mul_upd_tbb<ValueType>(tmp[3], cur_shape,
                                          m_minus(tmp[5], tmp[4]));
    });
    g.wait();
    std::vector<ValueType> r1 = m_plus(m_plus(d1, v1), d);
    std::vector<ValueType> r2 = m_plus(v1, h2);
    res = r1;
    for (size_t i = 0; i < res_shape[0] - r1.size(); i++) {
      res.push_back(r2[i]);
    }
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
  if (vec.size() != mat_shape[1]) {
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
