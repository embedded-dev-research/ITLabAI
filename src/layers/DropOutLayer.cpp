#include "layers/DropOutLayer.hpp"

#include <algorithm>
#include <functional>
#include <random>

namespace it_lab_ai {

void DropOutLayer::run(const Tensor &input, Tensor &output) {
  const double lower_bound = 0;
  const double upper_bound = 100;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  switch (input.get_type()) {
    case Type::kInt: {
      std::vector<int> vec = *input.as<int>();
      for (int &i : vec) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100) i = 0;
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    case Type::kFloat: {
      std::vector<float> vec = *input.as<float>();
      for (float &i : vec) {
        if (unif(rand_engine) < static_cast<float>(drop_rate_) * 100) i = 0;
      }
      output = make_tensor(vec, input.get_shape());
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
