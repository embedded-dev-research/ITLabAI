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
  } else if (axis_ != 0) {
    int start_dim = axis_;
    if (start_dim < 0) {
      start_dim += static_cast<int>(input_shape.dims());
    }

    if (start_dim < 0 || static_cast<size_t>(start_dim) >= input_shape.dims()) {
      throw std::runtime_error("FlattenLayer: Invalid axis value");
    }
    size_t flattened_size = 1;
    auto start_dim_size = static_cast<size_t>(start_dim);
    for (size_t i = start_dim_size; i < input_shape.dims(); ++i) {
      flattened_size *= input_shape[i];
    }
    if (start_dim > 0) {
      std::vector<size_t> dims;
      for (size_t i = 0; i < start_dim_size; ++i) {
        dims.push_back(input_shape[i]);
      }
      dims.push_back(flattened_size);
      output_shape = Shape(dims);
    } else {
      output_shape = Shape({flattened_size});
    }

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
