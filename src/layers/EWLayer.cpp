#include "layers/EWLayer.hpp"

namespace itlab_2023 {
void EWLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      EWLayerImpl<int> used_impl(input.get_shape(), func_, alpha_, beta_);
      output = make_tensor(used_impl.run(*input.as<int>()), input.get_shape());
      break;
    }
    case Type::kFloat: {
      EWLayerImpl<float> used_impl(input.get_shape(), func_, alpha_, beta_);
      output =
          make_tensor(used_impl.run(*input.as<float>()), input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
}  // namespace itlab_2023
