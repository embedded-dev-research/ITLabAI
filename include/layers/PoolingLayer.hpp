#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

#include "layers/Layer.hpp"

namespace itlab_2023 {

enum PoolingType : uint8_t { kAverage, kMax };

class PoolingLayer : public Layer {
 public:
  PoolingLayer() = default;
  PoolingLayer(const Shape& pooling_shape, std::string pooling_type = "average")
      : poolingShape_(pooling_shape), poolingType_(std::move(pooling_type)) {}
  static std::string get_name() { return "Pooling layer"; }
  void run(const Tensor& input, Tensor& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif

 private:
  Shape poolingShape_;
  std::string poolingType_;
};

inline bool isOutOfBounds(size_t index, int coord, const Shape& shape) {
  if (coord >= 0 && static_cast<size_t>(coord) < shape.dims()) {
    return (index >= shape[coord]);
  }
  return (index > 0);
}

template <typename ValueType>
ValueType avg_pooling(const std::vector<ValueType>& input) {
  if (input.empty()) {
    throw std::runtime_error("Empty input in avg pooling");
  }
  return std::accumulate(input.begin(), input.end(), ValueType(0)) /
         static_cast<ValueType>(input.size());
}

template <typename ValueType>
ValueType max_pooling(const std::vector<ValueType>& input) {
  if (input.empty()) {
    throw std::runtime_error("Empty input in max pooling");
  }
  return *(std::max_element(input.begin(), input.end()));
}

template <typename ValueType>
class PoolingLayerImpl : public LayerImpl<ValueType> {
 public:
  PoolingLayerImpl() = delete;
  PoolingLayerImpl(const Shape& input_shape, const Shape& pooling_shape,
                   const std::string& pooling_type = "average");
  PoolingLayerImpl(const PoolingLayerImpl& c) = default;
  PoolingLayerImpl& operator=(const PoolingLayerImpl& c) = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 private:
  Shape poolingShape_;
  PoolingType poolingType_;
};

template <typename ValueType>
PoolingLayerImpl<ValueType>::PoolingLayerImpl(const Shape& input_shape,
                                              const Shape& pooling_shape,
                                              const std::string& pooling_type)
    : LayerImpl<ValueType>(input_shape, input_shape),
      poolingShape_(pooling_shape) {
  if (input_shape.dims() > 4) {
    throw std::invalid_argument("Input dimensions is bigger than 4");
  }
  if (pooling_shape.dims() > input_shape.dims()) {
    throw std::invalid_argument("Pooling dims is bigger than the input dims");
  }
  if (pooling_shape.dims() > 2) {
    throw std::invalid_argument("Pooling dims is bigger than 2");
  }
  if (pooling_shape.dims() == 0) {
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
  size_t input_h_index = input_shape.dims() > 2 ? (input_shape.dims() - 2) : 0;
  for (size_t i = 0; i < pooling_shape.dims(); i++) {
    if (pooling_shape[i] == 0) {
      throw std::runtime_error("Zero division, pooling shape has zeroes");
    }
    this->outputShape_[input_h_index + i] =
        input_shape[input_h_index + i] / pooling_shape[i];
  }
}

template <typename ValueType>
std::vector<ValueType> PoolingLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit pooling layer");
  }
  std::vector<ValueType> pooling_buf;
  std::vector<ValueType> res;
  std::vector<size_t> coords;
  size_t tmpwidth = 0;
  size_t tmpheight = 0;
  int input_h_index = this->inputShape_.dims() > 2
                          ? (static_cast<int>(this->inputShape_.dims()) - 2)
                          : 0;
  // O(N^2)
  for (size_t n = 0; !isOutOfBounds(n, input_h_index - 2, this->outputShape_);
       n++) {
    for (size_t c = 0; !isOutOfBounds(c, input_h_index - 1, this->outputShape_);
         c++) {
      for (size_t i = 0; !isOutOfBounds(i, input_h_index, this->outputShape_);
           i++) {
        for (size_t j = 0;
             !isOutOfBounds(j, input_h_index + 1, this->outputShape_); j++) {
          tmpheight = poolingShape_[0] * i;
          if (poolingShape_.dims() == 1) {
            tmpwidth = j;
          } else {
            tmpwidth = poolingShape_[1] * j;
          }
          // to get matrix block for pooling
          for (size_t k = 0; !isOutOfBounds(k, 0, poolingShape_); k++) {
            for (size_t l = 0; !isOutOfBounds(l, 1, poolingShape_); l++) {
              if (this->inputShape_.dims() == 1) {
                pooling_buf.push_back(input[tmpheight + k]);
              } else {
                coords =
                    std::vector<size_t>({n, c, tmpheight + k, tmpwidth + l});
                pooling_buf.push_back(input[this->inputShape_.get_index(
                    std::vector<size_t>(coords.end() - this->inputShape_.dims(),
                                        coords.end()))]);
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
    }
  }
  return res;
}
}  // namespace itlab_2023
