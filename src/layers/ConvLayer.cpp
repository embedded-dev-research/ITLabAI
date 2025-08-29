#include "layers/ConvLayer.hpp"

namespace it_lab_ai {

void ConvolutionalLayer::run(const std::vector<Tensor>& input,
                             std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("ConvolutionalLayer: Input tensors not 1");
  }
  if (input[0].get_shape().dims() != 4) {
    throw std::out_of_range("input must be 4-dimensional");
  }
  switch (input[0].get_type()) {
    case Type::kInt: {
      if (kernel_.get_shape().dims() == 2) {
        if (dilations_ > 0) {
          dilations_--;
        }
        ConvImpl<int> used_impl(
            stride_, pads_, dilations_,
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 1]),
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 2]),
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 3]),
            input[0].get_shape()[input[0].get_shape().dims() - 1] *
                input[0].get_shape()[input[0].get_shape().dims() - 2],
            bias_.empty() ? std::vector<int>() : *bias_.as<int>());
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input[0].get_shape()[input[0].get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output[0] = make_tensor<int>(
            used_impl.run(
                *input[0].as<int>(),
                static_cast<int>(
                    input[0].get_shape()[input[0].get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input[0].get_shape()[input[0].get_shape().dims() - 2]) +
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
        switch (implType_) {
          case kSTL: {
            Conv4DSTL<int>(input[0], kernel_, bias_, output[0], stride_, pads_,
                           dilations_);
            break;
          }
          default: {
            Conv4D<int>(input[0], kernel_, bias_, output[0], stride_, pads_,
                        dilations_);
            break;
          }
        }
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
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 1]),
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 2]),
            static_cast<int>(
                input[0].get_shape()[input[0].get_shape().dims() - 3]),
            input[0].get_shape()[input[0].get_shape().dims() - 1] *
                input[0].get_shape()[input[0].get_shape().dims() - 2],
            bias_.empty() ? std::vector<float>() : *bias_.as<float>());
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input[0].get_shape()[input[0].get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output[0] = make_tensor<float>(
            used_impl.run(
                *input[0].as<float>(),
                static_cast<int>(
                    input[0].get_shape()[input[0].get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input[0].get_shape()[input[0].get_shape().dims() - 2]) +
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
        switch (implType_) {
          case kSTL: {
            Conv4DSTL<float>(input[0], kernel_, bias_, output[0], stride_,
                             pads_, dilations_);
            break;
          }
          default: {
            Conv4D<float>(input[0], kernel_, bias_, output[0], stride_, pads_,
                          dilations_);
            break;
          }
        }
      }
      break;
    }
    default: {
      throw std::runtime_error("Unsupported tensor type");
    }
  }
}

}  // namespace it_lab_ai
