#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

template <typename T>
void emplConv(const Tensor& input, Tensor& output, const Tensor& kernel_,
              size_t stride_, size_t pads_, size_t dilations_) {
  std::vector<T> startmatrix = *input.as<T>();
  int input_width =
      static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]);
  int input_height =
      static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]);
  int input_flow =
      static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]);
  int new_rows = input_width + 2 * static_cast<int>(pads_);
  int new_cols = input_height + 2 * static_cast<int>(pads_);
  std::vector<T> matrix(new_rows * new_cols * input_flow, 0);
  for (int i = 0; i < input_height; ++i) {
    for (int j = 0; j < input_width; ++j) {
      matrix[((i + pads_) * new_cols + j + pads_) * input_flow] =
          startmatrix[(i * input_width + j) * input_flow];
      matrix[((i + pads_) * new_cols + j + pads_) * input_flow + 1] =
          startmatrix[(i * input_width + j) * input_flow + 1];
      matrix[((i + pads_) * new_cols + j + pads_) * input_flow + 2] =
          startmatrix[(i * input_width + j) * input_flow + 2];
    }
  }
  size_t input_size = input.get_shape()[input.get_shape().dims() - 2] *
                      input.get_shape()[input.get_shape().dims() - 3];

  std::vector<T> startkernel = *kernel_.as<T>();
  size_t start_kernel_size =
      kernel_.get_shape()[kernel_.get_shape().dims() - 1];
  size_t kernel_size = (1 + start_kernel_size) * dilations_ + start_kernel_size;
  int center_distance = static_cast<int>((kernel_size - 1) / 2);
  std::vector<T> kernel(kernel_size * kernel_size, 0);
  for (int i = 0; i < start_kernel_size; i++) {
    for (int j = 0; j < start_kernel_size; j++) {
      kernel[(dilations_ + i) * kernel_size + j + (j + 1) * dilations_] =
          startkernel[i * start_kernel_size + j];
    }
  }
  std::vector<T> outputvec;
  for (int i = input_width + center_distance; i < static_cast<int>(input_size);
       i += static_cast<int>(stride_)) {
    for (int x = 0; x < 3; x++) {
      T color = 0;
      for (int coloms = -input_width; coloms < input_width + 1;
           coloms += input_width) {
        for (int str = -1; str < 2; str++) {
          auto kercol = static_cast<size_t>(coloms / input_width + 1);
          color += matrix[(i + coloms + str) * 3 + x] *
                   kernel[kercol * kernel_size + static_cast<size_t>(str + 1)];
        }
      }
      outputvec.push_back(color);
    }
    if ((i + center_distance + 1) % input_width == 0) {
      if (i + input_width + center_distance * 2 ==
          static_cast<int>(input_size)) {
        i += input_width + center_distance * 2 + 1;
      } else {
        i += input_width * (static_cast<int>(stride_) - 1) +
             (3 - static_cast<int>(stride_));
      }
    }
  }
  auto sizeforshape = static_cast<size_t>(
      ((input_width - 1 - static_cast<int>(kernel_size - 1)) /
       static_cast<int>(stride_)) +
      1);
  Shape sh({1, sizeforshape, sizeforshape, 3});
  output = make_tensor<T>(outputvec, sh);
}

class ConvolutionalLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;

 public:
  ConvolutionalLayer() = default;
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations) {
    stride_ = step;
    pads_ = pads;
    dilations_ = dilations;
  }
  void run(const Tensor& input, Tensor& output, const Tensor& kernel_);
};
void ConvolutionalLayer::run(const Tensor& input, Tensor& output,
                             const Tensor& kernel_) {
  switch (input.get_type()) {
    case Type::kInt: {
      emplConv<int>(input, output, kernel_, stride_, pads_, dilations_);
      break;
    }
    case Type::kFloat: {
      emplConv<float>(input, output, kernel_, stride_, pads_, dilations_);
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
