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

  this->validate_inputs(input);

  switch (input[0].get_type()) {
    case Type::kFloat:
      this->concatenate<float>(input, output[0]);
      break;
    case Type::kInt:
      this->concatenate<int>(input, output[0]);
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

std::vector<Tensor> ConcatLayer::reorderInputs(
    const std::vector<Tensor>& inputs) const {
  if (input_order_.empty() || input_order_.size() != inputs.size()) {
    return inputs;
  }

  std::vector<Tensor> reordered(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (input_order_[i] >= 0 &&
        static_cast<size_t>(input_order_[i]) < inputs.size()) {
      reordered[i] = inputs[input_order_[i]];
    } else {
      throw std::runtime_error("ConcatLayer: Invalid input order index");
    }
  }
  return reordered;
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

}  // namespace it_lab_ai