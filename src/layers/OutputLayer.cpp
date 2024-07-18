#include "layers/OutputLayer.hpp"

namespace itlab_2023 {

std::pair<std::vector<std::string>, Tensor> OutputLayer::top_k(
    const Tensor &input, size_t k) const {
  if (input.get_shape().dims() != 1) {
    throw std::invalid_argument("TopK function accepts only 1D vector");
  }
  std::vector<std::string> reslabels;
  Tensor resvector;
  switch (input.get_type()) {
    case Type::kFloat: {
      auto toppair = top_k_vec<float>(*input.as<float>(), labels_, k);
      reslabels = toppair.first;
      resvector = resvector = make_tensor(toppair.second);
      break;
    }
    case Type::kInt: {
      auto toppair = top_k_vec<int>(*input.as<int>(), labels_, k);
      reslabels = toppair.first;
      resvector = make_tensor(toppair.second);
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  };
  return make_pair(reslabels, resvector);
}
}  // namespace itlab_2023
