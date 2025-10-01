#include "layers/ReshapeLayer.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace it_lab_ai {

void ReshapeLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  if (input.empty()) {
    throw std::runtime_error("ReshapeLayer: At least 1 input tensor required");
  }

  const auto& data_tensor = input[0];
  std::vector<int64_t> target_shape = shape_;

  if (input.size() >= 2 && input[1].get_type() == Type::kInt) {
    const auto* shape_data = input[1].as<int64_t>();
    if (shape_data && !shape_data->empty()) {
      target_shape.assign(shape_data->begin(), shape_data->end());
    }
  }

  auto final_shape =
      calculate_output_shape(data_tensor.get_shape(), target_shape);

  switch (data_tensor.get_type()) {
    case Type::kFloat:
      reshape_impl<float>(data_tensor, output[0], final_shape);
      break;
    case Type::kInt:
      reshape_impl<int>(data_tensor, output[0], final_shape);
      break;
    default:
      throw std::runtime_error("Unsupported tensor data type for Reshape");
  }
}

std::vector<int64_t> ReshapeLayer::calculate_output_shape(
    const Shape& input_shape,
    const std::vector<int64_t>& requested_shape){
  size_t total_elements = 1;
  for (size_t i = 0; i < input_shape.dims(); ++i) {
    total_elements *= input_shape[i];
  }

  std::vector<int64_t> output_shape;
  output_shape.reserve(requested_shape.size());

  int negative_dim = -1;
  size_t inferred_size = total_elements;

  for (size_t i = 0; i < requested_shape.size(); ++i) {
    int64_t dim = requested_shape[i];

    if (dim == -1) {
      if (negative_dim != -1) {
        throw std::runtime_error("Reshape: Only one dimension can be -1");
      }
      negative_dim = static_cast<int>(i);
      output_shape.push_back(1);
    } else if (dim == 0) {
      if (i >= input_shape.dims()) {
        throw std::runtime_error("Reshape: Dimension 0 index out of range");
      }
      auto dim_value = static_cast<int64_t>(input_shape[i]);
      output_shape.push_back(dim_value);
      if (dim_value != 0) {
        inferred_size /= static_cast<size_t>(dim_value);
      }
    } else {
      if (dim < 0 && dim != -1) {
        throw std::runtime_error(
            "Reshape: Negative dimension value not supported");
      }
      output_shape.push_back(dim);
      if (dim != 0) {
        inferred_size /= static_cast<size_t>(dim);
      }
    }
  }

  if (negative_dim != -1) {
    output_shape[negative_dim] = static_cast<int64_t>(inferred_size);
  }

  size_t new_total = 1;
  for (int64_t dim : output_shape) {
    new_total *= static_cast<size_t>(dim);
  }

  if (new_total != total_elements) {
    throw std::runtime_error("Reshape: Total elements mismatch");
  }

  return output_shape;
}

template <typename T>
void ReshapeLayer::reshape_impl(
    const Tensor& input, Tensor& output,
    const std::vector<int64_t>& target_shape) const {
  const auto* input_data = input.as<T>();
  if (!input_data) {
    throw std::runtime_error("Reshape: Invalid input data");
  }

  std::vector<size_t> shape_size_t;
  shape_size_t.reserve(target_shape.size());
  for (int64_t dim : target_shape) {
    shape_size_t.push_back(static_cast<size_t>(dim));
  }

  Shape new_shape(shape_size_t);
  output = make_tensor(*input_data, new_shape);
}

template void ReshapeLayer::reshape_impl<float>(
    const Tensor&, Tensor&, const std::vector<int64_t>&) const;
template void ReshapeLayer::reshape_impl<int>(
    const Tensor&, Tensor&, const std::vector<int64_t>&) const;

}  // namespace it_lab_ai