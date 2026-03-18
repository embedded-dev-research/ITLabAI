#include "layers/PoolingLayer.hpp"

namespace it_lab_ai {

void PoolingLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void PoolingLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output,
                       const RuntimeOptions& options) {
  if (input.size() != 1) {
    throw std::runtime_error("PoolingLayer: Input tensors not 1");
  }
  ParBackend backend = options.par_backend;
  switch (input[0].get_type()) {
    case Type::kInt: {
      PoolingLayerImpl<int> used_impl(input[0].get_shape(), poolingShape_,
                                      strides_, pads_, dilations_, ceil_mode_,
                                      poolingType_, backend);
      output[0] = make_tensor(used_impl.run(*input[0].as<int>()),
                              used_impl.get_output_shape());
      break;
    }
    case Type::kFloat: {
      PoolingLayerImpl<float> used_impl(input[0].get_shape(), poolingShape_,
                                        strides_, pads_, dilations_, ceil_mode_,
                                        poolingType_, backend);
      output[0] = make_tensor(used_impl.run(*input[0].as<float>()),
                              used_impl.get_output_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
