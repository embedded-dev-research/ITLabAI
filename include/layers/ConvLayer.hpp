#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class ConvolutionalLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  Tensor kernel_;

 public:
  ConvolutionalLayer() = default;
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations,
                     const Tensor& kernel) {
    stride_ = step;
    pads_ = pads;
    dilations_ = dilations;
    kernel_ = kernel;
  }
  void run(const Tensor& input, Tensor& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return kernel_; }
#endif
};

template <typename ValueType>
class ConvImpl : public LayerImpl<ValueType> {
 private:
  int input_width_;
  int input_height_;
  int input_flow_;
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  size_t input_size_;

 public:
  ConvImpl() = delete;
  ConvImpl(size_t stride, size_t pads, size_t dilations, int input_width,
           int input_height, int input_flow, size_t input_size)
      : input_width_(input_width),
        input_height_(input_height),
        input_flow_(input_flow),
        stride_(stride),
        pads_(pads),
        dilations_(dilations),
        input_size_(input_size) {}
  ConvImpl(const ConvImpl& c) = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override {
    return input;
  }
  std::vector<ValueType> run(std::vector<ValueType> startmatrix, int new_rows,
                             int new_cols, std::vector<ValueType> startkernel,
                             size_t start_kernel_size, size_t kernel_size,
                             int center_distance) const {
    std::vector<ValueType> matrix(new_rows * new_cols * input_flow_, 0);
    for (int i = 0; i < input_height_; ++i) {
      for (int j = 0; j < input_width_; ++j) {
        matrix[((i + pads_) * new_cols + j + pads_) * input_flow_] =
            startmatrix[(i * input_width_ + j) * input_flow_];
        matrix[((i + pads_) * new_cols + j + pads_) * input_flow_ + 1] =
            startmatrix[(i * input_width_ + j) * input_flow_ + 1];
        matrix[((i + pads_) * new_cols + j + pads_) * input_flow_ + 2] =
            startmatrix[(i * input_width_ + j) * input_flow_ + 2];
      }
    }
    std::vector<ValueType> kernel(kernel_size * kernel_size, 0);
    for (int i = 0; i < static_cast<int>(start_kernel_size); i++) {
      for (int j = 0; j < static_cast<int>(start_kernel_size); j++) {
        kernel[(dilations_ + i) * static_cast<int>(kernel_size) + j +
               (j + 1) * dilations_] =
            startkernel[i * static_cast<int>(start_kernel_size) + j];
      }
    }
    std::vector<ValueType> outputvec;
    for (int i = input_width_ + center_distance;
         i < static_cast<int>(input_size_); i += static_cast<int>(stride_)) {
      for (int x = 0; x < 3; x++) {
        ValueType color = 0;
        for (int coloms = -input_width_; coloms < input_width_ + 1;
             coloms += input_width_) {
          for (int str = -1; str < 2; str++) {
            if (input_width_ == 0) {
              throw std::out_of_range("Input = 0");
            }
            auto kercol = static_cast<size_t>(coloms / input_width_ + 1);
            color +=
                matrix[(i + coloms + str) * 3 + x] *
                kernel[kercol * kernel_size + static_cast<size_t>(str + 1)];
          }
        }
        outputvec.push_back(color);
      }
      if ((i + center_distance + 1) % input_width_ == 0) {
        if (i + input_width_ + center_distance * 2 ==
            static_cast<int>(input_size_)) {
          i += input_width_ + center_distance * 2 + 1;
        } else {
          i += input_width_ * (static_cast<int>(stride_) - 1) +
               (3 - static_cast<int>(stride_));
        }
      }
    }
    return outputvec;
  }
};
}  // namespace itlab_2023
