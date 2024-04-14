#include "layers/ConvLayer.hpp"
void ConvolutionalLayer::run(const Tensor& input, Tensor& output,
                             const Tensor& kernel_) const{
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