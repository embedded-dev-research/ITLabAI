#include "layers/SoftmaxLayer.hpp"

#include <numeric>

namespace it_lab_ai {

void SoftmaxLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("SoftmaxLayer: Exactly 1 input tensor required");
  }

  switch (input[0].get_type()) {
    case Type::kFloat:
      softmax_impl<float>(input[0], output[0]);
      break;
    case Type::kInt:
      softmax_int_impl(input[0], output[0]);
      break;
    default:
      throw std::runtime_error("SoftmaxLayer: Unsupported tensor type");
  }
}

template <typename T>
void SoftmaxLayer::softmax_impl(const Tensor& input, Tensor& output) const {
  const auto* input_data = input.as<T>();
  if (!input_data) {
    throw std::runtime_error("Softmax: Invalid input data");
  }

  const auto& shape = input.get_shape();
  size_t normalized_axis = normalize_axis(shape, axis_);

  size_t outer_size = 1;
  for (size_t i = 0; i < normalized_axis; ++i) {
    outer_size *= shape[i];
  }

  size_t axis_size = shape[normalized_axis];

  size_t inner_size = 1;
  for (size_t i = normalized_axis + 1; i < shape.dims(); ++i) {
    inner_size *= shape[i];
  }

  std::vector<T> output_data(input_data->size());

  for (size_t outer = 0; outer < outer_size; ++outer) {
    for (size_t inner = 0; inner < inner_size; ++inner) {
      T max_val = std::numeric_limits<T>::lowest();
      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        if ((*input_data)[index] > max_val) {
          max_val = (*input_data)[index];
        }
      }

      T sum = T(0);
      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        T exp_val = std::exp((*input_data)[index] - max_val);
        output_data[index] = exp_val;
        sum += exp_val;
      }

      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        output_data[index] /= sum;
      }
    }
  }

  output = make_tensor(output_data, shape);
}

void SoftmaxLayer::softmax_int_impl(const Tensor& input, Tensor& output) const {
  const auto* input_data = input.as<int>();
  if (!input_data) {
    throw std::runtime_error("Softmax: Invalid input data");
  }

  const auto& shape = input.get_shape();
  size_t normalized_axis = normalize_axis(shape, axis_);

  size_t outer_size = 1;
  for (size_t i = 0; i < normalized_axis; ++i) {
    outer_size *= shape[i];
  }

  size_t axis_size = shape[normalized_axis];

  size_t inner_size = 1;
  for (size_t i = normalized_axis + 1; i < shape.dims(); ++i) {
    inner_size *= shape[i];
  }

  std::vector<float> float_output_data(input_data->size());

  for (size_t outer = 0; outer < outer_size; ++outer) {
    for (size_t inner = 0; inner < inner_size; ++inner) {
      int max_val = std::numeric_limits<int>::min();
      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        if ((*input_data)[index] > max_val) {
          max_val = (*input_data)[index];
        }
      }

      float sum = 0.0F;
      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        float exp_val =
            std::exp(static_cast<float>((*input_data)[index] - max_val));
        float_output_data[index] = exp_val;
        sum += exp_val;
      }

      for (size_t axis = 0; axis < axis_size; ++axis) {
        size_t index =
            outer * axis_size * inner_size + axis * inner_size + inner;
        float_output_data[index] /= sum;
      }
    }
  }

  std::vector<int> int_output_data(input_data->size());
  for (size_t i = 0; i < input_data->size(); ++i) {
    int_output_data[i] = static_cast<int>(float_output_data[i] * 1000);
  }

  output = make_tensor(int_output_data, shape);
}

size_t SoftmaxLayer::normalize_axis(const Shape& shape, int axis) {
  size_t rank = shape.dims();
  if (axis < 0) {
    axis = static_cast<int>(rank) + axis;
  }
  if (axis < 0 || static_cast<size_t>(axis) >= rank) {
    throw std::runtime_error("Softmax: Invalid axis value");
  }
  return static_cast<size_t>(axis);
}

template void SoftmaxLayer::softmax_impl<float>(const Tensor&, Tensor&) const;

}  // namespace it_lab_ai