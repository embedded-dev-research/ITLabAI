#include "layers/FCLayer.hpp"

namespace it_lab_ai {

void FCLayer::run(const std::vector<Tensor>& input,
                  std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("FCLayer: Input tensors not 1");
  }
  if (input[0].get_type() != weights_.get_type()) {
    throw std::invalid_argument("input[0] and weights data type aren't same");
  }
  if (bias_.get_type() != weights_.get_type()) {
    throw std::invalid_argument("Bias and weights data type aren't same");
  }
  switch (input[0].get_type()) {
    case Type::kInt: {
      FCLayerImpl<int> used_impl(*weights_.as<int>(), weights_.get_shape(),
                                 *bias_.as<int>());
      output[0] =
          make_tensor(used_impl.run(*input[0].as<int>()),
                      {(*input[0].as<int>()).size() / weights_.get_shape()[1] *
                       weights_.get_shape()[0]});
      break;
    }
    case Type::kFloat: {
      FCLayerImpl<float> used_impl(*weights_.as<float>(), weights_.get_shape(),
                                   *bias_.as<float>());
      output[0] =
          make_tensor(used_impl.run(*input[0].as<float>()),
                      {(*input[0].as<float>()).size() /
                       weights_.get_shape()[1] * weights_.get_shape()[0]});
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
