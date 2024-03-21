#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

template <typename ValueType>
class ConvolutionalLayer : public Layer<ValueType> {
 public:
  ConvolutionalLayer() = delete;
  ConvolutionalLayer(const std::vector<ValueType>& kernel,
                     const Shape& kernel_shape, size_t image_width,
                     size_t image_height, int step);
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override;

 private:
  std::vector<ValueType> kernel_;
  Shape kernel_shape_;
  size_t image_width_;
  size_t image_height_;
  int step_;
};

template <typename ValueType>
ConvolutionalLayer<ValueType>::ConvolutionalLayer(
    const std::vector<ValueType>& kernel, const Shape& kernel_shape,
    size_t image_width, size_t image_height, int step)
    : Layer<ValueType>(Shape({image_height, image_width}),
                       Shape({image_height - kernel_shape[0] - (step - 1),
                              image_width - kernel_shape[1] - (step - 1)})),
      kernel_(kernel),
      kernel_shape_(kernel_shape),
      image_width_(image_width),
      image_height_(image_height),
      step_(step) {}

template <typename ValueType>
std::vector<ValueType> ConvolutionalLayer<ValueType>::run(
    const std::vector<ValueType>& input) const {
  size_t input_size = input.size();
  size_t kernelSize = static_cast<int>(std::sqrt(kernel_.size()));
  size_t output_height = this->outputShape_[0];
  size_t output_width = this->outputShape_[1];
  int input_width = static_cast<int>(std::sqrt(input_size / 3));
  if (input_size != image_width_ * image_height_ * 3) {
    throw std::invalid_argument("Input size doesn't fit ConvolutionalLayer");
  }
  std::vector<ValueType> output;
  for (int i = input_width + 1; i < static_cast<int>(input_size / 3);
       i += step_) {
    for (int x = 0; x < 3; x++) {
      ValueType color = 0;
      int ckernel = 0;
      for (int coloms = -input_width; coloms < input_width + 1;
           coloms += input_width) {
        for (int str = -1; str < 2; str++) {
          color += input[(i + coloms + str) * 3 + x] * kernel_[ckernel++];
        }
      }
      output.push_back(color);
    }
    if ((i + static_cast<int>((kernelSize - 1) / 2) + 1) % input_width == 0) {
      if (i + input_width + static_cast<int>((kernelSize - 1)) ==
          static_cast<int>(input_size / 3)) {
        i += input_width + static_cast<int>((kernelSize - 1)) + 1;
      } else {
        i += input_width * (step_ - 1) + (3 - step_);
      }
    }
  }
  if (output.size() != output_height * output_width * 3) {
    throw std::invalid_argument("Output size doesn't fit ConvolutionalLayer");
  }
  return output;
}
