#include "layers/SplitLayer.hpp"

namespace it_lab_ai {

void SplitLayer::run(const Tensor& input, Tensor& output) { output = input; }

void SplitLayer::run(const Tensor& input, std::vector<Tensor>& outputs) {
  validate(input);
  const auto& shape = input.get_shape();
  const int axis = get_normalized_axis(static_cast<int>(shape.dims()));

  std::vector<int> part_sizes;
  if (!splits_.empty()) {
    part_sizes = splits_;
  } else {
    const int base_size = static_cast<int>(shape[axis]) / num_outputs_;
    const int remainder = static_cast<int>(shape[axis]) % num_outputs_;
    part_sizes.assign(num_outputs_, base_size);
    if (remainder > 0) {
      part_sizes.back() += remainder;
    }
  }

  outputs.clear();
  for (int size : part_sizes) {
    Shape out_shape = shape;
    out_shape[axis] = static_cast<size_t>(size);
    outputs.emplace_back(out_shape, input.get_type());
  }

  switch (input.get_type()) {
    case Type::kFloat:
      split_impl<float>(input, outputs);
      break;
    case Type::kInt:
      split_impl<int>(input, outputs);
      break;
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
}

template <typename T>
void SplitLayer::split_impl(const Tensor& input,
                            std::vector<Tensor>& outputs) const {
  const auto& input_data = *input.as<T>();
  const Shape& shape = input.get_shape();
  const int axis = get_normalized_axis(static_cast<int>(shape.dims()));
  const auto& part_sizes =
      splits_.empty()
          ? std::vector<int>(num_outputs_,
                             static_cast<int>(shape[axis]) / num_outputs_)
          : splits_;

  size_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= shape[i];
  }

  size_t inner_size = 1;
  for (size_t i = axis + 1; i < shape.dims(); ++i) {
    inner_size *= shape[i];
  }

  size_t input_offset = 0;
  for (auto& output : outputs) {
    auto& output_data = *output.as<T>();
    const size_t output_axis_size = output.get_shape()[axis];

    for (size_t outer = 0; outer < outer_size; ++outer) {
      for (size_t a = 0; a < output_axis_size; ++a) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
          size_t input_pos = outer * shape[axis] * inner_size +
                             (input_offset + a) * inner_size + inner;
          size_t output_pos =
              outer * output_axis_size * inner_size + a * inner_size + inner;
          output_data[output_pos] = input_data[input_pos];
        }
      }
    }
    input_offset += output_axis_size;
  }
}

void SplitLayer::validate(const Tensor& input) const {
  if (input.get_shape().dims() == 0) {
    throw std::runtime_error("SplitLayer: Cannot split scalar tensor");
  }

  const int axis =
      get_normalized_axis(static_cast<int>(input.get_shape().dims()));
  const size_t axis_size = input.get_shape()[axis];

  if (!splits_.empty()) {
    int sum = 0;
    for (int s : splits_) {
      if (s <= 0) throw std::runtime_error("Split size must be positive");
      sum += s;
    }
    if (sum != static_cast<int>(axis_size)) {
      throw std::runtime_error("Sum of splits must match axis size");
    }
  } else if (num_outputs_ <= 0) {
    throw std::runtime_error("num_outputs must be positive");
  }
}

int SplitLayer::get_normalized_axis(int rank) const {
  if (axis_ < 0) return axis_ + rank;
  if (axis_ >= rank) throw std::runtime_error("Axis out of bounds");
  return axis_;
}

template void SplitLayer::split_impl<float>(const Tensor&,
                                            std::vector<Tensor>&) const;
template void SplitLayer::split_impl<int>(const Tensor&,
                                          std::vector<Tensor>&) const;

}  // namespace it_lab_ai