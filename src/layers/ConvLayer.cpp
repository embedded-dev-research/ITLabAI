#include "layers/ConvLayer.hpp"

namespace itlab_2023 {

void ConvolutionalLayer::run(const Tensor& input, Tensor& output) {
  if (input.get_shape().dims() != 4) {
    throw std::out_of_range("Input must be 4-dimensional");
  }
  switch (input.get_type()) {
    case Type::kInt: {
      if (kernel_.get_shape().dims() == 2) {
        if (dilations_ > 0) {
          dilations_--;
        }
        ConvImpl<int> used_impl(
            stride_, pads_, dilations_,
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
            input.get_shape()[input.get_shape().dims() - 1] *
                input.get_shape()[input.get_shape().dims() - 2],
            bias_.empty() ? std::vector<int>() : *bias_.as<int>());
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output = make_tensor<int>(
            used_impl.run(
                *input.as<int>(),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 2]) +
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
      } else {
        Conv4D<int>(input, kernel_, bias_, output, stride_, pads_, dilations_);
      }
      break;
    }
    case Type::kFloat: {
      if (kernel_.get_shape().dims() == 2) {
        if (dilations_ > 0) {
          dilations_--;
        }
        ConvImpl<float> used_impl(
            stride_, pads_, dilations_,
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
            static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
            input.get_shape()[input.get_shape().dims() - 1] *
                input.get_shape()[input.get_shape().dims() - 2],
            bias_.empty() ? std::vector<float>() : *bias_.as<float>());
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output = make_tensor<float>(
            used_impl.run(
                *input.as<float>(),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 2]) +
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
      } else {
        Conv4D<float>(input, kernel_, bias_, output, stride_, pads_,
                      dilations_);
      }
      break;
    }
    default: {
      throw std::runtime_error("Unsupported tensor type");
    }
  }
}

}  // namespace itlab_2023