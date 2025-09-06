#include "layers/TransposeLayer.hpp"

#include <algorithm>
#include <numeric>

namespace it_lab_ai {

void TransposeLayer::run(const std::vector<Tensor>& input,
                         std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("TransposeLayer: Input tensors not 1");
  }

  const auto& shape = input[0].get_shape();

  std::vector<int64_t> perm = perm_;
  if (perm.empty()) {
    perm.resize(shape.dims());
    std::iota(perm.begin(), perm.end(), 0);
  }

  validate_perm(shape, perm);

  switch (input[0].get_type()) {
    case Type::kFloat:
      transpose_impl<float>(input[0], output[0], perm);
      break;
    case Type::kInt:
      transpose_impl<int>(input[0], output[0], perm);
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
  new_dims.reserve(shape.dims());
  for (const auto& axis : perm) {
    new_dims.push_back(shape[static_cast<size_t>(axis)]);
  }
  Shape new_shape(new_dims);

  std::vector<size_t> input_strides(shape.dims());
  size_t stride = 1;
  for (size_t dim = shape.dims(); dim-- > 0;) {
    input_strides[dim] = stride;
    stride *= shape[dim];
  }

  std::vector<size_t> output_strides(new_shape.dims());
  stride = 1;
  for (size_t dim = new_shape.dims(); dim-- > 0;) {
    output_strides[dim] = stride;
    stride *= new_shape[dim];
  }

  std::vector<T> output_values(input_data->size());

  if (shape.dims() == 2) {
    const size_t rows = shape[0];
    const size_t cols = shape[1];

    if (perm[0] == 1 && perm[1] == 0) {
      for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
          const size_t new_index = j * rows + i;
          if (new_index >= output_values.size()) {
            throw std::runtime_error(
                "Index out of bounds during 2D transposition");
          }
          output_values[new_index] = (*input_data)[i * cols + j];
        }
      }
    } else {
      for (size_t i = 0; i < input_data->size(); ++i) {
        size_t old_index = i;
        size_t new_index = 0;

        for (size_t dim = 0; dim < perm.size(); ++dim) {
          const auto axis = static_cast<size_t>(perm[dim]);
          const size_t coord = (old_index / input_strides[axis]) % shape[axis];
          new_index += coord * output_strides[dim];
        }

        if (new_index >= output_values.size()) {
          throw std::runtime_error(
              "Index out of bounds during 2D transposition");
        }
        output_values[new_index] = (*input_data)[i];
      }
    }
  } else {
    for (size_t i = 0; i < input_data->size(); ++i) {
      size_t old_index = i;
      size_t new_index = 0;

      for (size_t dim = 0; dim < perm.size(); ++dim) {
        const auto axis = static_cast<size_t>(perm[dim]);
        const size_t coord = (old_index / input_strides[axis]) % shape[axis];
        new_index += coord * output_strides[dim];
      }

      if (new_index >= output_values.size()) {
        throw std::runtime_error("Index out of bounds during transposition");
      }
      output_values[new_index] = (*input_data)[i];
    }
  }

  output = make_tensor(output_values, new_shape);
}

void TransposeLayer::validate_perm(const Shape& input_shape,
                                   const std::vector<int64_t>& perm) {
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