#include "layers/EWLayer.hpp"

namespace it_lab_ai {

void EWLayer::run(const std::vector<Tensor>& input,
                  std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("EWLayer: Input tensors not 1");
  }
  switch (input[0].get_type()) {
    case Type::kInt: {
      EWLayerImpl<int> used_impl(input[0].get_shape(), func_, alpha_, beta_,
                                 type_parall_);
      std::vector<int> tmp = used_impl.run(*input[0].as<int>());
      auto start = std::chrono::high_resolution_clock::now();
      output[0] = make_tensor(tmp, input[0].get_shape());
      auto end = std::chrono::high_resolution_clock::now();
      auto total_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      std::cout << total_duration.count() << std::endl;
      break;
    }
    case Type::kFloat: {
      EWLayerImpl<float> used_impl(input[0].get_shape(), func_, alpha_, beta_,
                                   type_parall_);
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
