#include "layers/PoolingLayer.hpp"

namespace it_lab_ai {

void PoolingLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("PoolingLayer: Input tensors not 1");
  }

  switch (input[0].get_type()) {
    case Type::kInt: {
      switch (implType_) {
        case kTBB: {
          PoolingLayerImplTBB<int> used_impl(
              input[0].get_shape(), poolingShape_, strides_, pads_, dilations_,
              ceil_mode_, poolingType_);
          output[0] = make_tensor(used_impl.run(*input[0].as<int>()),
                                  used_impl.get_output_shape());
          break;
        }
        default: {
          PoolingLayerImpl<int> used_impl(input[0].get_shape(), poolingShape_,
                                          strides_, pads_, dilations_,
                                          ceil_mode_, poolingType_);
          output[0] = make_tensor(used_impl.run(*input[0].as<int>()),
                                  used_impl.get_output_shape());
          break;
        }
      }
      break;
    }
    case Type::kFloat: {
      switch (implType_) {
        case kTBB: {
          PoolingLayerImplTBB<float> used_impl(
              input[0].get_shape(), poolingShape_, strides_, pads_, dilations_,
              ceil_mode_, poolingType_);
          output[0] = make_tensor(used_impl.run(*input[0].as<float>()),
                                  used_impl.get_output_shape());
          break;
        }
        default: {
          PoolingLayerImpl<float> used_impl(input[0].get_shape(), poolingShape_,
                                            strides_, pads_, dilations_,
                                            ceil_mode_, poolingType_);
          output[0] = make_tensor(used_impl.run(*input[0].as<float>()),
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

}  // namespace it_lab_ai