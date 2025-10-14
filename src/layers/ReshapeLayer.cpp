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
      reshape_impl<float>(data_tensor, output[0], target_shape, final_shape);
      break;
    case Type::kInt:
      reshape_impl<int>(data_tensor, output[0], target_shape, final_shape);
      break;
    default:
      throw std::runtime_error("Unsupported tensor data type for Reshape");
  }
}

std::vector<int64_t> ReshapeLayer::calculate_output_shape(
    const Shape& input_shape, const std::vector<int64_t>& requested_shape) {
  std::vector<int64_t> target_shape = requested_shape;
  if (requested_shape[0] == 1 && input_shape[0] > 1) {
    target_shape[0] = static_cast<int64_t>(input_shape[0]);
  }

  size_t total_elements = 1;
  for (size_t i = 0; i < input_shape.dims(); ++i) {
    total_elements *= input_shape[i];
  }

  std::vector<int64_t> output_shape;
  output_shape.reserve(target_shape.size());

  int negative_dim = -1;
  size_t inferred_size = total_elements;

  for (size_t i = 0; i < target_shape.size(); ++i) {
    int64_t dim = target_shape[i];

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
      output_shape.push_back(dim);
      if (dim != 0) {
        inferred_size /= static_cast<size_t>(dim);
      }
    }
  }

  if (negative_dim != -1) {
    if (inferred_size == 0 ||
        inferred_size > std::numeric_limits<size_t>::max() / 1000) {
      throw std::runtime_error("Reshape: Invalid inferred dimension size");
    }
    output_shape[negative_dim] = static_cast<int64_t>(inferred_size);
  }

  return output_shape;
}

template <typename T>
void ReshapeLayer::reshape_impl(const Tensor& input, Tensor& output,
                                const std::vector<int64_t>& target_shape,
                                const std::vector<int64_t>& final_shape) const {
  const auto* input_data = input.as<T>();
  const Shape& input_shape = input.get_shape();

  if (input_shape[0] > 1 && target_shape[0] == 1) {
    apply_per_batch_reshape<T>(input, output, target_shape);
  } else {
    std::vector<size_t> shape_size_t;
    shape_size_t.reserve(final_shape.size());
    for (int64_t dim : final_shape) {
      shape_size_t.push_back(static_cast<size_t>(dim));
    }
    output = make_tensor(*input_data, Shape(shape_size_t));
  }
}

template <typename T>
void ReshapeLayer::apply_per_batch_reshape(
    const Tensor& input, Tensor& output,
    const std::vector<int64_t>& target_shape) const {
  const auto* input_data = input.as<T>();
  const Shape& input_shape = input.get_shape();
  size_t batch_size = input_shape[0];
  size_t elements_per_batch = input_shape.count() / batch_size;
  std::vector<int64_t> per_batch_target = target_shape;
  per_batch_target[0] = 1;

  Shape single_batch_input_shape = input_shape;
  single_batch_input_shape[0] = 1;

  std::vector<int64_t> single_batch_output_shape =
      calculate_output_shape(single_batch_input_shape, per_batch_target);

  std::vector<size_t> final_output_shape_size_t;
  final_output_shape_size_t.reserve(single_batch_output_shape.size());
  final_output_shape_size_t.push_back(batch_size);
  for (size_t i = 1; i < single_batch_output_shape.size(); ++i) {
    final_output_shape_size_t.push_back(
        static_cast<size_t>(single_batch_output_shape[i]));
  }

  Shape final_output_shape(final_output_shape_size_t);

  size_t output_elements_per_batch = final_output_shape.count() / batch_size;

  if (elements_per_batch != output_elements_per_batch) {
    throw std::runtime_error("Reshape: Per-batch elements mismatch");
  }

  std::vector<T> output_data(final_output_shape.count());

  for (size_t b = 0; b < batch_size; ++b) {
    size_t input_offset = b * elements_per_batch;
    size_t output_offset = b * output_elements_per_batch;

    for (size_t i = 0; i < elements_per_batch; ++i) {
      output_data[output_offset + i] = (*input_data)[input_offset + i];
    }
  }

  output = make_tensor(output_data, final_output_shape);
}

template void ReshapeLayer::reshape_impl<float>(
    const Tensor&, Tensor&, const std::vector<int64_t>&,
    const std::vector<int64_t>&) const;
template void ReshapeLayer::reshape_impl<int>(
    const Tensor&, Tensor&, const std::vector<int64_t>&,
    const std::vector<int64_t>&) const;

}  // namespace it_lab_ai