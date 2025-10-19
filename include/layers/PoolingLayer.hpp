#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"

namespace it_lab_ai {

enum PoolingType : uint8_t { kAverage, kMax };

class PoolingLayer : public Layer {
 public:
  PoolingLayer(const Shape& pooling_shape, const Shape& strides = {2, 2},
               const Shape& pads = {0, 0, 0, 0},
               const Shape& dilations = {1, 1}, bool ceil_mode = false,
               std::string pooling_type = "average",
               ImplType implType = kDefault)
      : Layer(kPooling),
        poolingShape_(pooling_shape),
        strides_(strides),
        pads_(pads),
        dilations_(dilations),
        ceil_mode_(ceil_mode),
        poolingType_(std::move(pooling_type)),
        implType_(implType) {}
  PoolingLayer(const Shape& pooling_shape, std::string pooling_type = "average",
               ImplType implType = kDefault)
      : Layer(kPooling),
        poolingShape_(pooling_shape),
        strides_({2, 2}),
        pads_({0, 0, 0, 0}),
        dilations_({1, 1}),
        ceil_mode_(false),
        poolingType_(std::move(pooling_type)),
        implType_(implType) {}
  void setStrides(size_t h, size_t w) { strides_ = {h, w}; }
  void setPads(size_t top, size_t bottom, size_t left, size_t right) {
    pads_ = {top, bottom, left, right};
  }
  void setDilations(size_t h, size_t w) { dilations_ = {h, w}; }
  void setCeilMode(bool ceil_mode) { ceil_mode_ = ceil_mode; }
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
  Shape strides_;
  Shape pads_;
  Shape dilations_;
  bool ceil_mode_;
  std::string poolingType_;
  ImplType implType_;
};

inline size_t coord_size(size_t coord, const Shape& shape) {
  if (coord < shape.dims()) {
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
                   const std::string& pooling_type = "average")
      : PoolingLayerImpl(input_shape, pooling_shape, {2, 2}, {0, 0, 0, 0},
                         {1, 1}, false, pooling_type) {}
  PoolingLayerImpl(const Shape& input_shape, const Shape& pooling_shape,
                   const Shape& strides = {2, 2},
                   const Shape& pads = {0, 0, 0, 0},
                   const Shape& dilations = {1, 1}, bool ceil_mode = false,
                   const std::string& pooling_type = "average");
  PoolingLayerImpl(const PoolingLayerImpl& c) = default;
  PoolingLayerImpl& operator=(const PoolingLayerImpl& c) = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 protected:
  Shape poolingShape_;
  Shape strides_;
  Shape pads_;
  Shape dilations_;
  bool ceil_mode_;
  PoolingType poolingType_;
};

template <typename ValueType>
PoolingLayerImpl<ValueType>::PoolingLayerImpl(
    const Shape& input_shape, const Shape& pooling_shape, const Shape& strides,
    const Shape& pads, const Shape& dilations, bool ceil_mode,
    const std::string& pooling_type)
    : LayerImpl<ValueType>(input_shape, input_shape),
      poolingShape_(pooling_shape),
      strides_(strides),
      pads_(pads),
      dilations_(dilations),
      ceil_mode_(ceil_mode),
      poolingType_(kAverage) {
  if (pooling_shape[0] == 0 && pooling_shape[1] == 0) {
    poolingShape_ = Shape({input_shape[input_shape.dims() - 2],
                           input_shape[input_shape.dims() - 1]});
    strides_ = Shape({1, 1});
    pads_ = Shape({0, 0, 0, 0});
    dilations_ = Shape({1, 1});
    this->outputShape_ = input_shape;
    for (size_t i = 2; i < input_shape.dims(); ++i) {
      this->outputShape_[i] = 1;
    }
    poolingType_ = kAverage;
    return;
  }
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
    std::cerr << "ERROR: Unknown pooling type: '" << pooling_type << "'"
              << std::endl;
    throw std::invalid_argument("Pooling type " + pooling_type +
                                " is not supported");
  }
  this->outputShape_ = input_shape;
  for (size_t i = 0; i < pooling_shape.dims(); i++) {
    size_t input_size =
        input_shape[input_shape.dims() - pooling_shape.dims() + i];
    size_t kernel_size = pooling_shape[i];
    size_t stride = strides[i];
    size_t pad = pads[i] + pads[pooling_shape.dims() + i];
    size_t dilation = dilations[i];

    size_t effective_kernel_size = (kernel_size - 1) * dilation + 1;

    size_t output_size;
    if (ceil_mode) {
      output_size = static_cast<size_t>(
                        std::ceil((input_size + pad - effective_kernel_size) /
                                  static_cast<float>(stride))) +
                    1;
    } else {
      output_size = static_cast<size_t>(
                        std::floor((input_size + pad - effective_kernel_size) /
                                   static_cast<float>(stride))) +
                    1;
    }

    this->outputShape_[input_shape.dims() - pooling_shape.dims() + i] =
        output_size;
  }
}

template <typename ValueType>
std::vector<ValueType> PoolingLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit pooling layer");
  }

  std::vector<ValueType> res(this->outputShape_.count(), ValueType(0));

  size_t spatial_dims = poolingShape_.dims();
  int batch_dim = this->inputShape_.dims() > spatial_dims ? 0 : -1;
  int channel_dim = this->inputShape_.dims() > spatial_dims + 1 ? 1 : -1;

  for (size_t n = 0; n < (batch_dim >= 0 ? this->outputShape_[batch_dim] : 1);
       n++) {
    for (size_t c = 0;
         c < (channel_dim >= 0 ? this->outputShape_[channel_dim] : 1); c++) {
      for (size_t h = 0;
           h < this->outputShape_[this->outputShape_.dims() - spatial_dims];
           h++) {
        for (size_t w = 0;
             w < (spatial_dims > 1
                      ? this->outputShape_[this->outputShape_.dims() -
                                           spatial_dims + 1]
                      : 1);
             w++) {
          std::vector<ValueType> pooling_buf;

          int start_h =
              static_cast<int>(h * strides_[0]) - static_cast<int>(pads_[0]);
          int start_w = spatial_dims > 1 ? static_cast<int>(w * strides_[1]) -
                                               static_cast<int>(pads_[2])
                                         : 0;

          for (size_t kh = 0; kh < poolingShape_[0]; kh++) {
            for (size_t kw = 0; kw < (spatial_dims > 1 ? poolingShape_[1] : 1);
                 kw++) {
              int pos_h = start_h + static_cast<int>(kh * dilations_[0]);
              int pos_w = spatial_dims > 1
                              ? start_w + static_cast<int>(kw * dilations_[1])
                              : 0;

              if (pos_h >= 0 &&
                  pos_h < static_cast<int>(
                              this->inputShape_[this->inputShape_.dims() -
                                                spatial_dims]) &&
                  (spatial_dims <= 1 ||
                   (pos_w >= 0 &&
                    pos_w < static_cast<int>(
                                this->inputShape_[this->inputShape_.dims() -
                                                  spatial_dims + 1])))) {
                std::vector<size_t> input_coords(this->inputShape_.dims(), 0);
                if (batch_dim >= 0) input_coords[batch_dim] = n;
                if (channel_dim >= 0) input_coords[channel_dim] = c;
                input_coords[this->inputShape_.dims() - spatial_dims] = pos_h;
                if (spatial_dims > 1)
                  input_coords[this->inputShape_.dims() - spatial_dims + 1] =
                      pos_w;

                size_t input_index = this->inputShape_.get_index(input_coords);
                pooling_buf.push_back(input[input_index]);
              }
            }
          }

          std::vector<size_t> output_coords(this->outputShape_.dims(), 0);
          if (batch_dim >= 0) output_coords[batch_dim] = n;
          if (channel_dim >= 0) output_coords[channel_dim] = c;
          output_coords[this->outputShape_.dims() - spatial_dims] = h;
          if (spatial_dims > 1)
            output_coords[this->outputShape_.dims() - spatial_dims + 1] = w;

          size_t output_index = this->outputShape_.get_index(output_coords);

          if (!pooling_buf.empty()) {
            switch (this->poolingType_) {
              case kAverage:
                res[output_index] = avg_pooling(pooling_buf);
                break;
              case kMax:
                res[output_index] = max_pooling(pooling_buf);
                break;
              default:
                throw std::runtime_error("Unknown pooling type");
            }
          }
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
                      const Shape& strides = {2, 2},
                      const Shape& pads = {0, 0, 0, 0},
                      const Shape& dilations = {1, 1}, bool ceil_mode = false,
                      const std::string& pooling_type = "average")
      : PoolingLayerImpl<ValueType>(input_shape, pooling_shape, strides, pads,
                                    dilations, ceil_mode, pooling_type) {}
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;
};

template <typename ValueType>
std::vector<ValueType> PoolingLayerImplTBB<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit pooling layer");
  }

  std::vector<ValueType> res(this->outputShape_.count(), ValueType(0));

  size_t spatial_dims = this->poolingShape_.dims();
  int batch_dim = this->inputShape_.dims() > spatial_dims ? 0 : -1;
  int channel_dim = this->inputShape_.dims() > spatial_dims + 1 ? 1 : -1;

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(
          0, batch_dim >= 0 ? this->outputShape_[batch_dim] : 1),
      [&](const oneapi::tbb::blocked_range<size_t>& r1) {
        for (size_t n = r1.begin(); n < r1.end(); n++) {
          oneapi::tbb::parallel_for(
              oneapi::tbb::blocked_range<size_t>(
                  0, channel_dim >= 0 ? this->outputShape_[channel_dim] : 1),
              [&](const oneapi::tbb::blocked_range<size_t>& r2) {
                for (size_t c = r2.begin(); c < r2.end(); c++) {
                  for (size_t h = 0;
                       h < this->outputShape_[this->outputShape_.dims() -
                                              spatial_dims];
                       h++) {
                    for (size_t w = 0;
                         w <
                         (spatial_dims > 1
                              ? this->outputShape_[this->outputShape_.dims() -
                                                   spatial_dims + 1]
                              : 1);
                         w++) {
                      std::vector<ValueType> pooling_buf;

                      int start_h = static_cast<int>(h * this->strides_[0]) -
                                    static_cast<int>(this->pads_[0]);
                      int start_w =
                          spatial_dims > 1
                              ? static_cast<int>(w * this->strides_[1]) -
                                    static_cast<int>(this->pads_[2])
                              : 0;

                      for (size_t kh = 0; kh < this->poolingShape_[0]; kh++) {
                        for (size_t kw = 0;
                             kw <
                             (spatial_dims > 1 ? this->poolingShape_[1] : 1);
                             kw++) {
                          int pos_h = start_h + static_cast<int>(
                                                    kh * this->dilations_[0]);
                          int pos_w =
                              spatial_dims > 1
                                  ? start_w + static_cast<int>(
                                                  kw * this->dilations_[1])
                                  : 0;

                          if (pos_h >= 0 &&
                              pos_h < static_cast<int>(
                                          this->inputShape_[this->inputShape_
                                                                .dims() -
                                                            spatial_dims]) &&
                              (spatial_dims <= 1 ||
                               (pos_w >= 0 &&
                                pos_w < static_cast<int>(
                                            this->inputShape_
                                                [this->inputShape_.dims() -
                                                 spatial_dims + 1])))) {
                            std::vector<size_t> input_coords(
                                this->inputShape_.dims(), 0);
                            if (batch_dim >= 0) input_coords[batch_dim] = n;
                            if (channel_dim >= 0) input_coords[channel_dim] = c;
                            input_coords[this->inputShape_.dims() -
                                         spatial_dims] = pos_h;
                            if (spatial_dims > 1)
                              input_coords[this->inputShape_.dims() -
                                           spatial_dims + 1] = pos_w;

                            size_t input_index =
                                this->inputShape_.get_index(input_coords);
                            pooling_buf.push_back(input[input_index]);
                          }
                        }
                      }

                      std::vector<size_t> output_coords(
                          this->outputShape_.dims(), 0);
                      if (batch_dim >= 0) output_coords[batch_dim] = n;
                      if (channel_dim >= 0) output_coords[channel_dim] = c;
                      output_coords[this->outputShape_.dims() - spatial_dims] =
                          h;
                      if (spatial_dims > 1)
                        output_coords[this->outputShape_.dims() - spatial_dims +
                                      1] = w;

                      size_t output_index =
                          this->outputShape_.get_index(output_coords);

                      if (!pooling_buf.empty()) {
                        switch (this->poolingType_) {
                          case kAverage:
                            res[output_index] = avg_pooling(pooling_buf);
                            break;
                          case kMax:
                            res[output_index] = max_pooling(pooling_buf);
                            break;
                          default:
                            throw std::runtime_error("Unknown pooling type");
                        }
                      }
                    }
                  }
                }
              });
        }
      });

  return res;
}

}  // namespace it_lab_ai