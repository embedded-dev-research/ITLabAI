#include "layers/PoolingLayer.hpp"

namespace itlab_2023 {

void PoolingLayer::run(const Tensor& input, Tensor& output) {
  switch (input.get_type()) {
    case Type::kFloat: {
      PoolingLayerImpl<float> used_impl(input.get_shape(), poolingShape_,
                                        poolingType_);
      output = make_tensor(used_impl.run(*input.as<float>()),
                           used_impl.get_output_shape());
      break;
    }
    case Type::kInt: {
      PoolingLayerImpl<int> used_impl(input.get_shape(), poolingShape_,
                                      poolingType_);
      output = make_tensor(used_impl.run(*input.as<int>()),
                           used_impl.get_output_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
}  // namespace itlab_2023
