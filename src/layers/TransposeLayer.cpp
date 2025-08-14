#include "layers/TransposeLayer.hpp"

#include <algorithm>
#include <numeric>

namespace it_lab_ai {

void TransposeLayer::run(const Tensor& input, Tensor& output) {
  const auto& shape = input.get_shape();

  std::vector<int64_t> perm = perm_;
  if (perm.empty()) {
    perm.resize(shape.dims());
    std::iota(perm.begin(), perm.end(), 0);
  }

  validate_perm(shape, perm);

  switch (input.get_type()) {
    case Type::kFloat:
      transpose_impl<float>(input, output, perm);
      break;
    case Type::kInt:
      transpose_impl<int>(input, output, perm);
      break;
    default:
      throw std::runtime_error("Unsupported tensor data type");
  }
}

template <typename T>
void TransposeLayer::transpose_impl(const Tensor& input, Tensor& output,
                                    const std::vector<int64_t>& perm) const {
  const auto& shape = input.get_shape();
  const auto* input_data = input.as<T>();

  if (!input_data || input_data->empty()) {
    throw std::runtime_error("Input tensor is empty or invalid");
  }

  std::vector<size_t> new_dims;
  for (const auto& axis : perm) {
    new_dims.push_back(shape[static_cast<size_t>(axis)]);
  }
  Shape new_shape(new_dims);

  std::vector<T> output_values(input_data->size());

  std::vector<size_t> new_indices(shape.dims());
  std::vector<size_t> old_indices(shape.dims());

  for (size_t i = 0; i < input_data->size(); ++i) {
    size_t remaining = i;
    for (size_t dim = shape.dims(); dim-- > 0;) {
      old_indices[dim] = remaining % shape[dim];
      remaining /= shape[dim];
    }

    for (size_t dim = 0; dim < perm.size(); ++dim) {
      new_indices[dim] = old_indices[static_cast<size_t>(perm[dim])];
    }

    size_t new_index = 0;
    size_t stride = 1;
    for (size_t dim = new_shape.dims(); dim-- > 0;) {
      new_index += new_indices[dim] * stride;
      stride *= new_shape[dim];
    }

    if (new_index >= output_values.size()) {
      throw std::runtime_error("Index out of bounds during transposition");
    }
    output_values[new_index] = (*input_data)[i];
  }

  output = make_tensor(output_values, new_shape);
}

void TransposeLayer::validate_perm(const Shape& input_shape,
                                   const std::vector<int64_t>& perm) const {
  if (perm.size() != input_shape.dims()) {
    throw std::invalid_argument("Permutation size must match input dimensions");
  }

  std::vector<bool> used_axes(input_shape.dims(), false);
  for (const auto& axis : perm) {
    if (axis < 0 || static_cast<size_t>(axis) >= input_shape.dims()) {
      throw std::invalid_argument("Invalid axis in permutation");
    }
    if (used_axes[static_cast<size_t>(axis)]) {
      throw std::invalid_argument("Duplicate axis in permutation");
    }
    used_axes[static_cast<size_t>(axis)] = true;
  }
}

template void TransposeLayer::transpose_impl<float>(
    const Tensor&, Tensor&, const std::vector<int64_t>&) const;
template void TransposeLayer::transpose_impl<int>(
    const Tensor&, Tensor&, const std::vector<int64_t>&) const;

}  // namespace it_lab_ai