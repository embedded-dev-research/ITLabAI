#include "layers/EWLayer.hpp"

#include <chrono>
#include <cmath>

namespace it_lab_ai {

void EWLayer::run(const std::vector<Tensor>& input,
                  std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void EWLayer::run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
                  const RuntimeOptions& options) {
  if (input.size() != 1) {
    throw std::runtime_error("EWLayer: Input tensors not 1");
  }

  ParBackend backend = options.par_backend;

  switch (input[0].get_type()) {
    case Type::kInt: {
      EWLayerImpl<int> used_impl(input[0].get_shape(), func_, alpha_, beta_,
                                 backend);
      std::vector<int> tmp = used_impl.run(*input[0].as<int>());
      output[0] = make_tensor(tmp, input[0].get_shape());
      break;
    }
    case Type::kFloat: {
      EWLayerImpl<float> used_impl(input[0].get_shape(), func_, alpha_, beta_,
                                   backend);
      output[0] = make_tensor(used_impl.run(*input[0].as<float>()),
                              input[0].get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
