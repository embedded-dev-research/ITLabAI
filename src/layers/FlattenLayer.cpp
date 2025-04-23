#include "layers/FlattenLayer.hpp"

namespace itlab_2023 {

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

void FlattenLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      if (input.get_shape().dims() == 4) {
        Flatten4D<int>(input, output, order_);
      } else {
        output =
            make_tensor(*input.as<int>(), Shape({input.get_shape().count()}));
      }
      break;
    }
    case Type::kFloat: {
      if (input.get_shape().dims() == 4) {
        Flatten4D<float>(input, output, order_);
      } else {
        output =
            make_tensor(*input.as<float>(), Shape({input.get_shape().count()}));
      }
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}

}  // namespace itlab_2023
