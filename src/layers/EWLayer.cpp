#include "layers/EWLayer.hpp"

void EWLayer::run(const Tensor &input, Tensor &output, const std::string &function) {
  switch (input.get_type()) {
    case Type::kInt: {
      EWLayerimpl<int> used_impl(input.get_shape(), function);
      output = make_tensor(used_impl.run(*input.as<int>()), input.get_shape());
      break;
    }
    case Type::kFloat: {
      EWLayerimpl<float> used_impl(input.get_shape(), function);
      output = make_tensor(used_impl.run(*input.as<float>()), input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
