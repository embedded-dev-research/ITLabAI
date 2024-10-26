#include "layers/DropOutLayer.hpp"

namespace itlab_2023 {

void DropOutLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      std::vector<int> vec = *input.as<int>();
      std::vector<float> vecres(vec.size());
      for (size_t i = 0; i < vec.size(); i++) {
        vecres[i] = vec[i] * (1 - static_cast<float>(drop_rate_));
      }
      output = make_tensor(vecres, input.get_shape());
      break;
    }
    case Type::kFloat: {
      std::vector<float> vec = *input.as<float>();
      for (size_t i = 0; i < vec.size(); i++) {
        vec[i] *= (1 - static_cast<float>(drop_rate_));
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace itlab_2023