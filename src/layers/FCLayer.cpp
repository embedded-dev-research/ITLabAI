#include "layers/FCLayer.hpp"

namespace itlab_2023 {

void FCLayer::run(const Tensor& input, Tensor& output) {
  if (input.get_type() != weights_.get_type()) {
    throw std::invalid_argument("Input and weights data type aren't same");
  }
  if (bias_.get_type() != weights_.get_type()) {
    throw std::invalid_argument("Bias and weights data type aren't same");
  }
  switch (input.get_type()) {
    case Type::kInt: {
      switch (implType_) {
        case kDefault: {
          FCLayerImpl<int> used_impl(*weights_.as<int>(), weights_.get_shape(),
                                     *bias_.as<int>());
          output = make_tensor(used_impl.run(*input.as<int>()),
                               used_impl.get_output_shape());
          break;
        }
        case kTBB: {
          FCLayerImplTBB<int> used_impl(*weights_.as<int>(),
                                        weights_.get_shape(), *bias_.as<int>());
          output = make_tensor(used_impl.run(*input.as<int>()),
                               used_impl.get_output_shape());
          break;
        }
      }
      break;
    }
    case Type::kFloat: {
      switch (implType_) {
        case kDefault: {
          FCLayerImpl<float> used_impl(
              *weights_.as<float>(), weights_.get_shape(), *bias_.as<float>());
          output = make_tensor(used_impl.run(*input.as<float>()),
                               used_impl.get_output_shape());
          break;
        }
        case kTBB: {
          FCLayerImplTBB<float> used_impl(
              *weights_.as<float>(), weights_.get_shape(), *bias_.as<float>());
          output = make_tensor(used_impl.run(*input.as<float>()),
                               used_impl.get_output_shape());
          break;
        }
      }
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
}  // namespace itlab_2023
