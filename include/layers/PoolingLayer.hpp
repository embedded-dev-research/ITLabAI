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
  PoolingLayer(const Shape& pooling_shape, const Shape& strides = {2, 2},
               const Shape& pads = {0, 0, 0, 0},
               const Shape& dilations = {1, 1}, bool ceil_mode = false,
               std::string pooling_type = "average",
               ImplType implType = kDefault)
      : poolingShape_(pooling_shape),
        strides_(strides),
        pads_(pads),
        dilations_(dilations),
        ceil_mode_(ceil_mode),
        poolingType_(std::move(pooling_type)),
        implType_(implType) {}
  PoolingLayer(const Shape& pooling_shape, std::string pooling_type = "average",
               ImplType implType = kDefault)
      : poolingShape_(pooling_shape),
        strides_({2, 2}),
        pads_({0, 0, 0, 0}),
        dilations_({1, 1}),
        ceil_mode_(false),
        poolingType_(std::move(pooling_type)),
        implType_(implType) {}
  static std::string get_name() { return "Pooling layer"; }
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
    : LayerImpl<ValueType>(input_shape,
                           input_shape),  // временно, потом исправим
      poolingShape_(pooling_shape),
      strides_(strides),
      pads_(pads),
      dilations_(dilations),
      ceil_mode_(ceil_mode) {
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

    // ‘ормула дл€ расчета выходного размера с учетом padding, stride, dilation
    size_t input_size = input_shape[input_h_index + i];
    size_t kernel_size = pooling_shape[i];
    size_t stride = strides[i];
    size_t padding = pads[i];  // берем только верхний/левый padding
    size_t dilation = dilations[i];

    // Ёффективный размер €дра с учетом dilation
    size_t effective_kernel_size = (kernel_size - 1) * dilation + 1;

    // –асчет выходного размера
    size_t output_size;
    if (ceil_mode) {
      output_size = static_cast<size_t>(std::ceil(
                        (input_size + 2 * padding - effective_kernel_size) /
                        static_cast<float>(stride))) +
                    1;
    } else {
      output_size = static_cast<size_t>(std::floor(
                        (input_size + 2 * padding - effective_kernel_size) /
                        static_cast<float>(stride))) +
                    1;
    }

    this->outputShape_[input_h_index + i] = output_size;
  }
}

template <typename ValueType>
std::vector<ValueType> PoolingLayerImpl<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (input.size() != this->inputShape_.count()) {
    throw std::invalid_argument("Input size doesn't fit pooling layer");
  }

  std::vector<ValueType> res(this->outputShape_.count());
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
          std::vector<ValueType> pooling_buf;

          // –ассчитываем начальные позиции с учетом stride и padding
          size_t start_h = i * strides_[0] - pads_[0];  // pads_[0] = top
          size_t start_w = j * strides_[1] - pads_[2];  // pads_[2] = left

          for (size_t k = 0; k < poolingShape_[0]; k++) {
            for (size_t l = 0; l < poolingShape_[1]; l++) {
              // –ассчитываем позиции с учетом dilation
              size_t pos_h = start_h + k * dilations_[0];
              size_t pos_w = start_w + l * dilations_[1];

              // ѕровер€ем границы с учетом padding
              if (pos_h >= 0 && pos_h < this->inputShape_[input_h_index] &&
                  pos_w >= 0 && pos_w < this->inputShape_[input_h_index + 1]) {
                std::vector<size_t> coords = {n, c, pos_h, pos_w};
                pooling_buf.push_back(input[this->inputShape_.get_index(
                    std::vector<size_t>(coords.end() - this->inputShape_.dims(),
                                        coords.end()))]);
              }
            }
          }

          // ѕримен€ем pooling только если есть данные
          if (!pooling_buf.empty()) {
            size_t output_index = this->outputShape_.get_index({n, c, i, j});
            switch (poolingType_) {
              case kAverage:
                res[output_index] = avg_pooling(pooling_buf);
                break;
              case kMax:
                res[output_index] = max_pooling(pooling_buf);
                break;
              default:
                throw std::runtime_error("Unknown pooling type");
            }
          } else {
            // ќбработка случа€ когда нет данных (можно установить 0 или другое
            // значение)
            size_t output_index = this->outputShape_.get_index({n, c, i, j});
            res[output_index] = ValueType(0);
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
