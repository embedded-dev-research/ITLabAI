#include "layers/SplitLayer.hpp"

#include <algorithm>
#include <cstring>

namespace it_lab_ai {

void SplitLayer::run(const std::vector<Tensor>& input,
                     std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("SplitLayer: Input tensors not 1");
  }

  validate(input[0]);

  switch (input[0].get_type()) {
    case Type::kFloat:
      split_impl<float>(input[0], output);
      break;
    case Type::kInt:
      split_impl<int>(input[0], output);
      break;
    default:
      throw std::runtime_error("Unsupported tensor data type");
  }
}

template <typename T>
void SplitLayer::split_impl(const Tensor& input,
                            std::vector<Tensor>& outputs) const {
  const auto& input_data = *input.as<T>();
  const Shape& shape = input.get_shape();
  const int axis = get_normalized_axis(static_cast<int>(shape.dims()));

  std::vector<int64_t> part_sizes;
  if (splits_) {
    part_sizes = *splits_;
  } else {
    const int total_size = static_cast<int>(shape[axis]);
    const int base_size = total_size / *num_outputs_;
    const int remainder = total_size % *num_outputs_;

    part_sizes.reserve(*num_outputs_);
    for (int64_t i = 0; i < *num_outputs_; ++i) {
      part_sizes.push_back(i < remainder ? base_size + 1 : base_size);
    }
  }

  size_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < shape.dims(); ++i) {
    inner_size *= shape[i];
  }

  const size_t input_axis_stride = shape[axis] * inner_size;

  outputs.clear();
  outputs.reserve(part_sizes.size());

  size_t input_offset = 0;
  for (const auto part_size : part_sizes) {
    const auto output_axis_size = static_cast<size_t>(part_size);

    std::vector<size_t> output_shape_vec(shape.dims());
    for (size_t i = 0; i < shape.dims(); ++i) {
      output_shape_vec[i] =
          (static_cast<int>(i) == axis) ? output_axis_size : shape[i];
    }
    Shape output_shape(output_shape_vec);

    outputs.emplace_back(output_shape, input.get_type());
    auto& output_data = *outputs.back().as<T>();

    const size_t output_part_size = output_axis_size * inner_size;

    for (size_t outer = 0; outer < outer_size; ++outer) {
      const T* input_start =
          &input_data[outer * input_axis_stride + input_offset * inner_size];
      T* output_start = &output_data[outer * output_part_size];

      std::copy_n(input_start, output_part_size, output_start);
    }

    input_offset += output_axis_size;
  }
}

void SplitLayer::validate(const Tensor& input) const {
  if (input.get_shape().dims() == 0) {
    throw std::runtime_error("Cannot split scalar tensor");
  }

  const int axis =
      get_normalized_axis(static_cast<int>(input.get_shape().dims()));
  const int axis_size = static_cast<int>(input.get_shape()[axis]);

  if (splits_) {
    int64_t sum = 0;
    for (int64_t s : *splits_) {
      if (s <= 0) throw std::runtime_error("Split size must be positive");
      sum += s;
    }
    if (sum != axis_size) {
      throw std::runtime_error("Sum of splits must match axis size");
    }
  } else {
    if (*num_outputs_ <= 0) {
      throw std::runtime_error("num_outputs must be positive");
    }
    if (*num_outputs_ > axis_size) {
      throw std::runtime_error("num_outputs (" + std::to_string(*num_outputs_) +
                               ") cannot be greater than axis size (" +
                               std::to_string(axis_size) + ")");
    }
  }
}

int SplitLayer::get_normalized_axis(int rank) const {
  if (axis_ < -rank || axis_ >= rank) {
    throw std::runtime_error("Axis out of bounds");
  }
  return (axis_ < 0) ? axis_ + rank : axis_;
}

template void SplitLayer::split_impl<float>(const Tensor&,
                                            std::vector<Tensor>&) const;
template void SplitLayer::split_impl<int>(const Tensor&,
                                          std::vector<Tensor>&) const;

}  // namespace it_lab_ai