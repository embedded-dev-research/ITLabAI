#include "layers/FlattenLayer.hpp"

namespace itlab_2023 {

void FlattenLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      output =
	  make_tensor(*input.as<int>(), Shape({input.get_shape().count()}));
      break;
    }
    case Type::kFloat: {
      output =
          make_tensor(*input.as<float>(), Shape({input.get_shape().count()}));
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace itlab_2023
