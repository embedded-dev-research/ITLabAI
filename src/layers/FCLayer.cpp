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
      FCLayerImpl<int> used_impl(*weights_.as<int>(), weights_.get_shape(),
                                 *bias_.as<int>());
      output = make_tensor(used_impl.run(*input.as<int>()),
                           {(*input.as<int>()).size() /
                            weights_.get_shape()[1] * weights_.get_shape()[0]});
      break;
    }
    case Type::kFloat: {
      FCLayerImpl<float> used_impl(*weights_.as<float>(), weights_.get_shape(),
                                   *bias_.as<float>());
      output = make_tensor(used_impl.run(*input.as<float>()),
                           {(*input.as<float>()).size() /
                            weights_.get_shape()[1] * weights_.get_shape()[0]});
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
}  // namespace itlab_2023
