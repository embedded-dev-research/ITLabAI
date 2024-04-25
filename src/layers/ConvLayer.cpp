#include "layers/ConvLayer.hpp"
void ConvolutionalLayer::run(const Tensor& input, Tensor& output,
                             const Tensor& kernel_) const {
  switch (input.get_type()) {
    case Type::kInt: {
      ConvImpl<int> used_impl(
          stride_, pads_, dilations_,
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
          input.get_shape()[input.get_shape().dims() - 2] *
              input.get_shape()[input.get_shape().dims() - 3]);
      if (input.get_shape().dims() != 4) {
        throw std::out_of_range("Input = 0");
      }
      auto sizeforshape = static_cast<size_t>(
          ((static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]) -
            1 -
            static_cast<int>(
                (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                    dilations_ +
                kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
           static_cast<int>(stride_)) +
          1);
      Shape sh({1, sizeforshape, sizeforshape, 3});
      output = make_tensor<int>(
          used_impl.run(*input.as<int>(),
                        static_cast<int>(
                            input.get_shape()[input.get_shape().dims() - 2]) +
                            2 * static_cast<int>(pads_),
                        static_cast<int>(
                            input.get_shape()[input.get_shape().dims() - 3]) +
                            2 * static_cast<int>(pads_),
                        *kernel_.as<int>(),
              kernel_.get_shape()[kernel_.get_shape().dims() - 1],
              (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1],
              static_cast<int>(
                  ((1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                       dilations_ +
                   kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1) /
                  2)),
          sh);
      break;
    }
    case Type::kFloat: {
      ConvImpl<float> used_impl(
          stride_, pads_, dilations_,
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
          input.get_shape()[input.get_shape().dims() - 2] *
              input.get_shape()[input.get_shape().dims() - 3]);
      if (input.get_shape().dims() != 4) {
        throw std::out_of_range("Input = 0");
      }
      auto sizeforshape = static_cast<size_t>(
          ((static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]) -
            1 -
            static_cast<int>(
                (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                    dilations_ +
                kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
           static_cast<int>(stride_)) +
          1);
      Shape sh({1, sizeforshape, sizeforshape, 3});
      output = make_tensor<float>(
          used_impl.run(
              *input.as<float>(),
              static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 2]) +
                  2 * static_cast<int>(pads_),
              static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 3]) +
                  2 * static_cast<int>(pads_),
              *kernel_.as<float>(),
              kernel_.get_shape()[kernel_.get_shape().dims() - 1],
              (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1],
              static_cast<int>(
                  ((1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                       dilations_ +
                   kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1) /
                  2)),
          sh);
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}