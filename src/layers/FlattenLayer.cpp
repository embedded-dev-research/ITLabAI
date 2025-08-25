#include "layers/FlattenLayer.hpp"

namespace it_lab_ai {

// reorder coords
std::vector<size_t> reorder(std::vector<size_t> order_vec,
                            std::vector<size_t> order) {
  size_t min_ind;
  for (size_t i = 0; i < order.size() - 1; i++) {
    min_ind = i;
    for (size_t j = i + 1; j < order.size(); j++) {
      if (order[j] < order[min_ind]) {
        min_ind = j;
      }
    }
    std::swap(order_vec[i], order_vec[min_ind]);
    std::swap(order[i], order[min_ind]);
  }
  return order_vec;
}

void FlattenLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  switch (input[0].get_type()) {
    case Type::kInt: {
      if (input[0].get_shape().dims() == 4) {
        Flatten4D<int>(input[0], output[0], order_);
      } else {
        output[0] = make_tensor(*input[0].as<int>(),
                                Shape({input[0].get_shape().count()}));
      }
      break;
    }
    case Type::kFloat: {
      if (input[0].get_shape().dims() == 4) {
        Flatten4D<float>(input[0], output[0], order_);
      } else {
        output[0] = make_tensor(*input[0].as<float>(),
                                Shape({input[0].get_shape().count()}));
      }
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace it_lab_ai
