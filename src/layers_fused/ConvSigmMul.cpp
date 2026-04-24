#include "layers/BinaryOpLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers_fused/ConvSigmMul.hpp"

namespace it_lab_ai {

void ConvSigmMulLayer::run(const std::vector<Tensor>& input,
                           std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void ConvSigmMulLayer::run(const std::vector<Tensor>& input,
                           std::vector<Tensor>& output,
                           const RuntimeOptions& options) {
  std::vector<Tensor> temp_output(2, Tensor());

  ConvolutionalLayer conv(stride_, pads_, dilations_, kernel_, bias_, group_,
                          useLegacyImpl_);

  conv.run(input, temp_output, options);

  temp_output[1] = temp_output[0];

  switch (input[0].get_type()) {
    case Type::kFloat: {
      sigmoid<float>(temp_output[1]);
      break;
    }
    case Type::kInt: {
      sigmoid<int>(temp_output[1]);
      break;
    }
    default: {
      throw std::runtime_error("Unsupported type for convsigmmul");
    }
  }

  BinaryOpLayer binaryop(BinaryOpLayer::Operation::kMul);
  binaryop.run(temp_output, output, options);
}

}  // namespace it_lab_ai
