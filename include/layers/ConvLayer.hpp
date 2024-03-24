#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

template <typename ValueType>
class ConvolutionalLayer : public Layer<ValueType> {
 public:
  ConvolutionalLayer() = default;
  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override {
    return input;
  };
  void run(const Tensor& input, Tensor& output,
           const std::vector<ValueType>& kernel, size_t step_);
};
template <typename ValueType>
void ConvolutionalLayer<ValueType>::run(const Tensor& input, Tensor& output,
                                        const std::vector<ValueType>& kernel_,
                                        size_t step_) {
  std::vector<ValueType> matrix = *input.as<ValueType>();
  size_t input_size = matrix.size();
  size_t kernel_size = static_cast<int>(std::sqrt(kernel_.size()));
  int input_width = static_cast<int>(std::sqrt(input_size / 3));
  std::vector<ValueType> outputvec;
  for (int i = input_width + static_cast<int>((kernel_size - 1) / 2);
       i < static_cast<int>(input_size / 3); i += static_cast<int>(step_)) {
    for (int x = 0; x < 3; x++) {
      ValueType color = 0;
      int ckernel = 0;
      for (int coloms = -input_width; coloms < input_width + 1;
           coloms += input_width) {
        for (int str = -1; str < 2; str++) {
          color += matrix[(i + coloms + str) * 3 + x] * kernel_[ckernel++];
        }
      }
      outputvec.push_back(color);
    }
    if ((i + static_cast<int>((kernel_size - 1) / 2) + 1) % input_width == 0) {
      if (i + input_width + static_cast<int>((kernel_size - 1)) ==
          static_cast<int>(input_size / 3)) {
        i += input_width + static_cast<int>((kernel_size - 1)) + 1;
      } else {
        i += input_width * (static_cast<int>(step_) - 1) +
             (3 - static_cast<int>(step_));
      }
    }
  }
  Shape sh({1,
            static_cast<size_t>(
                ((input_width - 1 - static_cast<int>(kernel_size - 1)) /
                 static_cast<int>(step_)) +
                1),
            static_cast<size_t>(
                ((input_width - 1 - static_cast<int>(kernel_size - 1)) /
                 static_cast<int>(step_)) +
                1),
            3});
  output = make_tensor<float>(outputvec, sh);
}