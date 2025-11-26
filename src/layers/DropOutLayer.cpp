#include "layers/DropOutLayer.hpp"

#include <algorithm>
#include <functional>
#include <random>

namespace it_lab_ai {

void DropOutLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("DropOutLayer: Input tensors not 1");
  }
  const double lower_bound = 0;
  const double upper_bound = 100;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  switch (input[0].get_type()) {
    case Type::kInt: {
      std::vector<int> vec = *input[0].as<int>();
      for (int& i : vec) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100) i = 0;
      }
      output[0] = make_tensor(vec, input[0].get_shape());
      break;
    }
    case Type::kFloat: {
      std::vector<float> vec = *input[0].as<float>();
      for (float& i : vec) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100) i = 0;
      }
      output[0] = make_tensor(vec, input[0].get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
