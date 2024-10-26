#include "layers/DropOutLayer.hpp"

#include <algorithm>
#include <functional>
#include <random>

namespace itlab_2023 {

void DropOutLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      std::vector<int> vec = *input.as<int>();
      std::random_device dev;
      std::mt19937 gen(dev());
      for (size_t i = 0; i < vec.size(); i++) {
        if (gen() % (101) < static_cast<float>(drop_rate_) * 100) vec[i] = 0;
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    case Type::kFloat: {
      std::vector<float> vec = *input.as<float>();
      std::random_device dev;
      std::mt19937 gen(dev());
      for (size_t i = 0; i < vec.size(); i++) {
        if (gen() % (101) < static_cast<float>(drop_rate_) * 100) vec[i] = 0;
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