#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer() = default;
  void run(const Tensor& input, Tensor& output, const Tensor& kernel_,
           size_t step_);
};
void ConvolutionalLayer::run(const Tensor& input, Tensor& output,
                             const Tensor& kernel_, size_t step_) {
  switch (input.get_type()) {
    case Type::kInt: {
      std::vector<int> matrix = *input.as<int>();
      size_t input_size = input.get_shape()[input.get_shape().dims() - 2] *
                          input.get_shape()[input.get_shape().dims() - 3];
      size_t kernel_size = kernel_.get_shape()[kernel_.get_shape().dims() - 1];
      int input_width =
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]);
      int centerDistance = static_cast<int>((kernel_size - 1) / 2);
      std::vector<int> outputvec;
      for (int i = input_width + centerDistance;
           i < static_cast<int>(input_size); i += static_cast<int>(step_)) {
        for (int x = 0; x < 3; x++) {
          int color = 0;
          for (int coloms = -input_width; coloms < input_width + 1;
               coloms += input_width) {
            for (int str = -1; str < 2; str++) {
              size_t kercol = static_cast<size_t>(coloms / input_width + 1);
              color += matrix[(i + coloms + str) * 3 + x] *
                       kernel_.get<int>({kercol, static_cast<size_t>(str + 1)});
            }
          }
          outputvec.push_back(color);
        }
        if ((i + centerDistance + 1) % input_width == 0) {
          if (i + input_width + centerDistance * 2 ==
              static_cast<int>(input_size)) {
            i += input_width + centerDistance * 2 + 1;
          } else {
            i += input_width * (static_cast<int>(step_) - 1) +
                 (3 - static_cast<int>(step_));
          }
        }
      }
      size_t sizeforshape = static_cast<size_t>(
          ((input_width - 1 - static_cast<int>(kernel_size - 1)) /
           static_cast<int>(step_)) +
          1);
      Shape sh({1, sizeforshape, sizeforshape, 3});
      output = make_tensor<int>(outputvec, sh);
      break;
    }
    case Type::kFloat: {
      std::vector<float> matrix = *input.as<float>();
      size_t input_size = input.get_shape()[input.get_shape().dims() - 2] *
                          input.get_shape()[input.get_shape().dims() - 3];
      size_t kernel_size = kernel_.get_shape()[kernel_.get_shape().dims() - 1];
      int input_width =
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]);
      int centerDistance = static_cast<int>((kernel_size - 1) / 2);
      std::vector<float> outputvec;
      for (int i = input_width + centerDistance;
           i < static_cast<int>(input_size); i += static_cast<int>(step_)) {
        for (int x = 0; x < 3; x++) {
          float color = 0;
          for (int coloms = -input_width; coloms < input_width + 1;
               coloms += input_width) {
            for (int str = -1; str < 2; str++) {
              size_t kercol = static_cast<size_t>(coloms / input_width + 1);
              color += matrix[(i + coloms + str) * 3 + x] *
                       kernel_.get<float>({kercol, static_cast<size_t>(str + 1)});
            }
          }
          outputvec.push_back(color);
        }
        if ((i + centerDistance + 1) % input_width == 0) {
          if (i + input_width + centerDistance * 2 ==
              static_cast<int>(input_size)) {
            i += input_width + centerDistance * 2 + 1;
          } else {
            i += input_width * (static_cast<int>(step_) - 1) +
                 (3 - static_cast<int>(step_));
          }
        }
      }
      size_t sizeforshape = static_cast<size_t>(
          ((input_width - 1 - static_cast<int>(kernel_size - 1)) /
           static_cast<int>(step_)) +
          1);
      Shape sh({1, sizeforshape, sizeforshape, 3});
      output = make_tensor<float>(outputvec, sh);
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}