#include "layers/FCLayer.hpp"

void FCLayer::run(const Tensor& input, Tensor& output, const Tensor& weights,
                  const Tensor& bias) {
  if (input.get_type() != weights.get_type()) {
    throw std::invalid_argument("Input and weights data type aren't same");
  }
  if (bias.get_type() != weights.get_type()) {
    throw std::invalid_argument("Bias and weights data type aren't same");
  }
  switch (input.get_type()) {
    case Type::kInt: {
      FCLayerimpl<int> used_impl(*weights.as<int>(), weights.get_shape(),
                                *bias.as<int>());
      output = make_tensor(used_impl.run(*input.as<int>()),
                           used_impl.get_output_shape());
      break;
    }
    case Type::kFloat: {
      FCLayerimpl<float> used_impl(*weights.as<float>(), weights.get_shape(),
                                  *bias.as<float>());
      output = make_tensor(used_impl.run(*input.as<float>()),
                           used_impl.get_output_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
