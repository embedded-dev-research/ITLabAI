#pragma once
#include <algorithm>

#include "layers/FCLayer.hpp"

template <typename ValueType>
class EWLayer : public Layer<ValueType> {
 public:
  EWLayer(const Shape& shape, ValueType (*unaryFunc)(const ValueType&)) {
    unaryFunc_ = unaryFunc;
    this->inputShape_ = shape;
    this->outputShape_ = shape;
  }
  EWLayer(const EWLayer& c) = default;
  EWLayer& operator=(const EWLayer& c) = default;
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

 private:
  ValueType (*unaryFunc_)(const ValueType&);
};

template <typename ValueType>
std::vector<ValueType> EWLayer<ValueType>::run(
    const std::vector<ValueType>& input) const {
  std::vector<ValueType> res(this->outputShape_.count());
  std::transform(input.begin(), input.end(), res.begin(), unaryFunc_);
  return res;
}
