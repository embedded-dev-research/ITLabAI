#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

#include "layers/Layer.hpp"

namespace it_lab_ai {

enum PoolingType : uint8_t { kAverage, kMax };

class PoolingLayer : public Layer {
 public:
  PoolingLayer() = default;
  PoolingLayer(const Shape& pooling_shape, std::string pooling_type = "average",
               ImplType implType = kDefault)
      : poolingShape_(pooling_shape),
        poolingType_(std::move(pooling_type)),
        implType_(implType) {}
  static std::string get_name() { return "Pooling layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
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
  ImplType implType_;
};

inline size_t coord_size(int coord, const Shape& shape) {
  if (coord >= 0 && static_cast<size_t>(coord) < shape.dims()) {
    return shape[coord];
  }
  return 1;
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

 protected:
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
  for (size_t n = 0; n < coord_size(input_h_index - 2, this->outputShape_);
       n++) {
    for (size_t c = 0; c < coord_size(input_h_index - 1, this->outputShape_);
         c++) {
      for (size_t i = 0; i < coord_size(input_h_index, this->outputShape_);
           i++) {
        for (size_t j = 0;
             j < coord_size(input_h_index + 1, this->outputShape_); j++) {
          tmpheight = poolingShape_[0] * i;
          if (poolingShape_.dims() == 1) {
            tmpwidth = j;
          } else {
            tmpwidth = poolingShape_[1] * j;
          }
          // to get matrix block for pooling
          for (size_t k = 0; k < coord_size(0, poolingShape_); k++) {
            for (size_t l = 0; l < coord_size(1, poolingShape_); l++) {
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

template <typename ValueType>
class PoolingLayerImplTBB : public PoolingLayerImpl<ValueType> {
 public:
  PoolingLayerImplTBB(const Shape& input_shape, const Shape& pooling_shape,
                      const std::string& pooling_type = "average")
      : PoolingLayerImpl<ValueType>(input_shape, pooling_shape, pooling_type) {}
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;
};

template <typename ValueType>
std::vector<ValueType> PoolingLayerImplTBB<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit pooling layer");
  }
  std::vector<ValueType> res(this->outputShape_.count());
  int input_h_index = this->inputShape_.dims() > 2
                          ? (static_cast<int>(this->inputShape_.dims()) - 2)
                          : 0;
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(
          0, coord_size(input_h_index - 2, this->outputShape_), 0,
          coord_size(input_h_index - 1, this->outputShape_)),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (size_t n = r.rows().begin(); n < r.rows().end(); n++) {
          for (size_t c = r.cols().begin(); c < r.cols().end(); c++) {
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range2d<size_t>(
                    0, coord_size(input_h_index, this->outputShape_), 0,
                    coord_size(input_h_index + 1, this->outputShape_)),
                [&](oneapi::tbb::blocked_range2d<size_t> r1) {
                  for (size_t i = r1.rows().begin(); i < r1.rows().end(); i++) {
                    for (size_t j = r1.cols().begin(); j < r1.cols().end();
                         j++) {
                      std::vector<ValueType> pooling_buf;
                      std::vector<size_t> coords;
                      size_t tmpwidth;
                      size_t tmpheight;
                      tmpheight = this->poolingShape_[0] * i;
                      if (this->poolingShape_.dims() == 1) {
                        tmpwidth = j;
                      } else {
                        tmpwidth = this->poolingShape_[1] * j;
                      }
                      for (size_t k = 0; k < coord_size(0, this->poolingShape_);
                           k++) {
                        for (size_t l = 0;
                             l < coord_size(1, this->poolingShape_); l++) {
                          if (this->inputShape_.dims() == 1) {
                            pooling_buf.push_back(input[tmpheight + k]);
                          } else {
                            coords = std::vector<size_t>(
                                {n, c, tmpheight + k, tmpwidth + l});
                            pooling_buf.push_back(
                                input[this->inputShape_.get_index(
                                    std::vector<size_t>(
                                        coords.end() - this->inputShape_.dims(),
                                        coords.end()))]);
                          }
                        }
                      }
                      coords = std::vector<size_t>({n, c, i, j});
                      switch (this->poolingType_) {
                        case kAverage:
                          if (this->inputShape_.dims() == 1) {
                            res[i] = avg_pooling(pooling_buf);
                          } else {
                            res[this->outputShape_.get_index(
                                std::vector<size_t>(
                                    coords.end() - this->inputShape_.dims(),
                                    coords.end()))] = avg_pooling(pooling_buf);
                          }
                          break;
                        case kMax:
                          if (this->inputShape_.dims() == 1) {
                            res[i] = max_pooling(pooling_buf);
                          } else {
                            res[this->outputShape_.get_index(
                                std::vector<size_t>(
                                    coords.end() - this->inputShape_.dims(),
                                    coords.end()))] = max_pooling(pooling_buf);
                            break;
                            default:
                              throw std::runtime_error("Unknown pooling type");
                          }
                      }
                    }
                  }
                });
          }
        }
      });
  return res;
}

}  // namespace it_lab_ai
