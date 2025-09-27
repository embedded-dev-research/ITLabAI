#include "layers/FlattenLayer.hpp"

namespace it_lab_ai {

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
  if (input.size() != 1) {
    throw std::runtime_error("FlattenLayer: Input tensors not 1");
  }
  const auto& input_tensor = input[0];
  const auto& input_shape = input_tensor.get_shape();
  Shape output_shape;

  if (!order_.empty() && order_.size() == 4) {
    switch (input_tensor.get_type()) {
      case Type::kFloat:
        Flatten4D<float>(input_tensor, output[0], order_);
        break;
      case Type::kInt:
        Flatten4D<int>(input_tensor, output[0], order_);
        break;
      default:
        throw std::runtime_error("Unsupported tensor type");
    }
  } else {
    size_t total_size = input_shape.count();
    output_shape = Shape({total_size});

    switch (input_tensor.get_type()) {
      case Type::kInt:
        output[0] = make_tensor(*input_tensor.as<int>(), output_shape);
        break;
      case Type::kFloat:
        output[0] = make_tensor(*input_tensor.as<float>(), output_shape);
        break;
      default:
        throw std::runtime_error("Unsupported tensor type");
    }
  }
}
}  // namespace it_lab_ai
