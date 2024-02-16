#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>

#include "layers/Layer.hpp"

template <typename ValueType>
ValueType avg_pooling(const std::vector<ValueType>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Zero division in avg pooling");
  }
  return std::accumulate(input.begin(), input.end(), ValueType(0)) /
         input.size();
}

template <typename ValueType>
ValueType max_pooling(const std::vector<ValueType>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Zero division in avg pooling");
  }
  return *(std::max_element(input.begin(), input.end()));
}

template <typename ValueType>
class PoolingLayer : public Layer<ValueType> {
 public:
  PoolingLayer() = delete;
  PoolingLayer(const Shape& input_shape, const Shape& pooling_shape,
               const std::string& pooling_type = "average");
  PoolingLayer(const PoolingLayer& c) = default;
  PoolingLayer& operator=(const PoolingLayer& c) = default;
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

 private:
  Shape poolingShape_;
  char poolingType_;
};

template <typename ValueType>
PoolingLayer<ValueType>::PoolingLayer(
    const Shape& input_shape, const Shape& pooling_shape,
    const std::string& pooling_type)
    : poolingShape_(pooling_shape), Layer<ValueType>(input_shape, input_shape) {
  if (pooling_shape.dims() > input_shape.dims()) {
    throw std::invalid_argument("Pooling dims is bigger than the input dims");
  }
  if (pooling_shape.dims() > 2 || pooling_shape.dims() == 0) {
    throw std::invalid_argument("Pooling dims is bigger than 2 or is 0");
  }
  if (pooling_type == "average") {
    poolingType_ = 0;
  } else if (pooling_type == "max") {
    poolingType_ = 1;
  } else {
    throw std::invalid_argument("No such pooling type");
  }
  for (size_t i = 0; i < pooling_shape.dims(); i++) {
    auto div_result = std::lldiv(input_shape[i], pooling_shape[i]);
    this->outputShape_[i] =
        div_result.rem > 0 ? (div_result.quot + 1) : div_result.quot;
  }
}

template <typename ValueType>
std::vector<ValueType> PoolingLayer<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit PoolingLayer");
  }
  std::vector<ValueType> pooling_buf;
  std::vector<ValueType> res;
  size_t tmpwidth = 0;
  size_t tmpheight = 0;
  size_t pshape1;
  // O(N^2)
  if (this->inputShape_.dims() != 1) {
    if (poolingShape_.dims() == 1) {
      pshape1 = 1;
    } else {
      pshape1 = poolingShape_[1];
    }
    for (size_t i = 0; i < this->outputShape_[0]; i++) {
      for (size_t j = 0; j < this->outputShape_[1]; j++) {
        tmpheight = poolingShape_[0] * i;
        tmpwidth = pshape1 * j;
        for (size_t k = 0; k < poolingShape_[0]; k++) {
          if (tmpheight + k >= this->inputShape_[0]) {
            continue;
          }
          for (size_t l = 0; l < pshape1; l++) {
            if (tmpwidth + l >= this->inputShape_[1]) {
              continue;
            }
            pooling_buf.push_back(input[this->inputShape_.get_index(
                {tmpheight + k, tmpwidth + l})]);
          }
        }
        switch (poolingType_) {
          case 0:
            res.push_back(avg_pooling(pooling_buf));
            break;
          case 1:
            res.push_back(max_pooling(pooling_buf));
            break;
          default:
            throw std::runtime_error("Unhandled exception");
        }
        pooling_buf.clear();
      }
    }
  } else {
    for (size_t i = 0; i < this->outputShape_[0]; i++) {
      tmpheight = poolingShape_[0] * i;
      for (size_t k = 0; k < poolingShape_[0]; k++) {
        if (tmpheight + k >= this->inputShape_[0]) {
          continue;
        }
        pooling_buf.push_back(
            input[this->inputShape_.get_index({tmpheight + k})]);
      }
      switch (poolingType_) {
        case 0:
          res.push_back(avg_pooling(pooling_buf));
          break;
        case 1:
          res.push_back(max_pooling(pooling_buf));
          break;
        default:
          throw std::runtime_error("Unhandled exception");
      }
      pooling_buf.clear();
    }
  }
  return res;
}
