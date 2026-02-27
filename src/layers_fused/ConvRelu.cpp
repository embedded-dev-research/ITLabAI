#include "layers_fused/ConvRelu.hpp"

#include "layers/ConvLayer.hpp"

namespace it_lab_ai {

void ConvReluLayer::run(const std::vector<Tensor>& input,
                        std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void ConvReluLayer::run(const std::vector<Tensor>& input,
                        std::vector<Tensor>& output,
                        const RuntimeOptions& options) {
  ConvolutionalLayer conv(stride_, pads_, dilations_, kernel_, bias_, group_,
                          useLegacyImpl_);
  conv.run(input, output, options);
  switch (input[0].get_type()) {
    case Type::kInt: {
      relu<int>(output[0]);
      break;
    }
    case Type::kFloat: {
      relu<float>(output[0]);
      break;
    }
    default: {
      throw std::runtime_error("Unsupported tensor type");
    }
  }
}

}  // namespace it_lab_ai
