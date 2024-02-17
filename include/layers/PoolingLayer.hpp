#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>

#include "layers/Layer.hpp"

enum PoolingType { kAverage, kMax };

inline bool isOutOfBounds(size_t index, size_t coord, const Shape& shape) {
  if (coord < shape.dims()) {
    return (index >= shape[coord]);
  }
  return (index >= 1);
}

template <typename ValueType>
ValueType avg_pooling(const std::vector<ValueType>& input) {
  if (input.empty()) {
    throw std::runtime_error("Empty input in avg pooling");
  }
  return std::accumulate(input.begin(), input.end(), ValueType(0)) /
         input.size();
}

template <typename ValueType>
ValueType max_pooling(const std::vector<ValueType>& input) {
  if (input.empty()) {
    throw std::runtime_error("Empty input in max pooling");
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
  PoolingType poolingType_;
};

template <typename ValueType>
PoolingLayer<ValueType>::PoolingLayer(const Shape& input_shape,
                                      const Shape& pooling_shape,
                                      const std::string& pooling_type)
    : poolingShape_(pooling_shape), Layer<ValueType>(input_shape, input_shape) {
  if (pooling_shape.dims() > input_shape.dims()) {
    throw std::invalid_argument("Pooling dims is bigger than the input dims");
  }
  if (pooling_shape.dims() > 2) {
    throw std::invalid_argument("Pooling dims is bigger than 2");
  } else if (pooling_shape.dims() == 0) {
    throw std::invalid_argument("Pooling shape has no dimensions");
  }
  if (pooling_type == "average") {
    poolingType_ = kAverage;
  } else if (pooling_type == "max") {
    poolingType_ = kMax;
  } else {
    throw std::invalid_argument("Pooling type " + pooling_type +
                                " is not supported");
  }
  for (size_t i = 0; i < pooling_shape.dims(); i++) {
    if (pooling_shape[i] == 0) {
      throw std::runtime_error("Zero division, pooling shape has zeroes");
    }
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
  // O(N^2)
  for (size_t i = 0; !isOutOfBounds(i, 0, this->outputShape_); i++) {
    for (size_t j = 0; !isOutOfBounds(j, 1, this->outputShape_); j++) {
      tmpheight = poolingShape_[0] * i;
      if (poolingShape_.dims() == 1) {
        tmpwidth = j;
      } else {
        tmpwidth = poolingShape_[1] * j;
      }
      // to get matrix block for pooling
      for (size_t k = 0; !isOutOfBounds(k, 0, poolingShape_); k++) {
        if (isOutOfBounds(tmpheight + k, 0, this->inputShape_)) {
          continue;
        }
        for (size_t l = 0; !isOutOfBounds(l, 1, poolingShape_); l++) {
          if (isOutOfBounds(tmpwidth + l, 1, this->inputShape_)) {
            continue;
          }
          if (this->inputShape_.dims() == 1) {
            pooling_buf.push_back(input[tmpheight + k]);
          } else {
            pooling_buf.push_back(input[this->inputShape_.get_index(
                {tmpheight + k, tmpwidth + l})]);
          }
        }
      }
      switch (poolingType_) {
        case kAverage:
          res.push_back(avg_pooling(pooling_buf));
          break;
        case kMax:
          res.push_back(max_pooling(pooling_buf));
          break;
        default:
          throw std::runtime_error("Unknown pooling type");
      }
      pooling_buf.clear();
    }
  }
  return res;
}
