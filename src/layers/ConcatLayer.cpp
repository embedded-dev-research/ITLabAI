#include "layers/ConcatLayer.hpp"

namespace it_lab_ai {

void ConcatLayer::run(const std::vector<Tensor>& input,
                      std::vector<Tensor>& output) {
  if (input.empty()) {
    throw std::runtime_error("ConcatLayer: No input tensors provided");
  }

  if (input.size() == 1) {
    output = input;
    return;
  }

  validate_inputs(input);

  switch (input[0].get_type()) {
    case Type::kFloat:
      concatenate<float>(input, output[0]);
      break;
    case Type::kInt:
      concatenate<int>(input, output[0]);
      break;
    default:
      throw std::runtime_error("ConcatLayer: Unsupported input tensor type");
  }
}

void ConcatLayer::validate_inputs(const std::vector<Tensor>& inputs) const {
  if (inputs.empty()) return;

  const Shape& first_shape = inputs[0].get_shape();
  Type first_type = inputs[0].get_type();
  const int64_t normalized_axis = normalize_axis(first_shape.dims());

  for (size_t i = 1; i < inputs.size(); ++i) {
    const Shape& shape = inputs[i].get_shape();
    if (shape.dims() != first_shape.dims()) {
      throw std::runtime_error(
          "ConcatLayer: All input tensors must have the same rank");
    }

    if (inputs[i].get_type() != first_type) {
      throw std::runtime_error(
          "ConcatLayer: All input tensors must have the same type");
    }

    for (size_t dim = 0; dim < shape.dims(); ++dim) {
      if (dim != static_cast<size_t>(normalized_axis) &&
          shape[dim] != first_shape[dim]) {
        throw std::runtime_error(
            "ConcatLayer: All input tensors must have the same shape except "
            "for the concatenation axis");
      }
    }
  }
}

int64_t ConcatLayer::normalize_axis(size_t rank) const {
  if (rank == 0) {
    throw std::runtime_error("ConcatLayer: Cannot concatenate scalar tensors");
  }

  int64_t axis = axis_;

  if (axis < 0) {
    axis += static_cast<int64_t>(rank);
  }

  if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
    throw std::runtime_error("ConcatLayer: Axis " + std::to_string(axis_) +
                             " out of range for tensor rank " +
                             std::to_string(rank));
  }

  return axis;
}

Shape ConcatLayer::calculate_output_shape(
    const std::vector<Tensor>& inputs) const {
  if (inputs.empty()) return Shape({});

  const Shape& first_shape = inputs[0].get_shape();
  std::vector<size_t> output_dims(first_shape.dims());
  for (size_t i = 0; i < first_shape.dims(); ++i) {
    output_dims[i] = first_shape[i];
  }

  const int64_t normalized_axis = normalize_axis(first_shape.dims());
  output_dims[normalized_axis] = 0;
  for (const auto& input : inputs) {
    output_dims[normalized_axis] += input.get_shape()[normalized_axis];
  }

  return Shape(output_dims);
}

template <typename T>
void ConcatLayer::concatenate(const std::vector<Tensor>& inputs,
                              Tensor& output) const {
  Shape output_shape = calculate_output_shape(inputs);
  std::vector<T> output_data(output_shape.count(), 0);

  const int64_t axis = normalize_axis(inputs[0].get_shape().dims());
  const size_t outer_size = [&]() {
    size_t size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      size *= output_shape[i];
    }
    return size;
  }();

  const size_t inner_size = [&]() {
    size_t size = 1;
    for (size_t i = axis + 1; i < output_shape.dims(); ++i) {
      size *= output_shape[i];
    }
    return size;
  }();

  size_t output_offset = 0;

  for (const auto& input : inputs) {
    const auto& input_data = *input.as<T>();
    const Shape& input_shape = input.get_shape();
    const size_t input_axis_size = input_shape[axis];

    for (size_t outer = 0; outer < outer_size; ++outer) {
      for (size_t a = 0; a < input_axis_size; ++a) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
          size_t input_pos =
              outer * input_axis_size * inner_size + a * inner_size + inner;

          size_t output_pos = outer * output_shape[axis] * inner_size +
                              (output_offset + a) * inner_size + inner;

          output_data[output_pos] = input_data[input_pos];
        }
      }
    }

    output_offset += input_axis_size;
  }

  output = make_tensor(output_data, output_shape);
}

template void ConcatLayer::concatenate<float>(const std::vector<Tensor>&,
                                              Tensor&) const;
template void ConcatLayer::concatenate<int>(const std::vector<Tensor>&,
                                            Tensor&) const;

}  // namespace it_lab_ai