#include "layers/ReduceSumLayer.hpp"

#include <algorithm>
#include <numeric>

namespace itlab_2023 {

ReduceSumLayer::ReduceSumLayer(int64_t keepdims) : keepdims_(keepdims) {}

void ReduceSumLayer::normalize_axes(const Shape& input_shape,
                                    std::vector<int64_t>& axes) const {
  const int64_t rank = static_cast<int64_t>(input_shape.dims());

  if (rank == 0) {
    if (!axes.empty()) {
      throw std::runtime_error("ReduceSum: Axis specified for scalar input");
    }
    return;
  }

  if (axes.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
    return;
  }

  for (auto& axis : axes) {
    if (axis < -rank || axis >= rank) {
      throw std::runtime_error("ReduceSum: Axis out of range");
    }
    if (axis < 0) axis += rank;
  }

  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
}

Shape ReduceSumLayer::calculate_output_shape(
    const Shape& input_shape, const std::vector<int64_t>& axes) const {
  if (input_shape.dims() == 0) {
    return Shape({});
  }

  std::vector<size_t> new_dims;

  if (keepdims_) {
    for (size_t i = 0; i < input_shape.dims(); ++i) {
      new_dims.push_back(input_shape[i]);
    }

    for (int64_t axis : axes) {
      if (axis >= 0 && axis < static_cast<int64_t>(new_dims.size())) {
        new_dims[axis] = 1;
      }
    }
  } else {
    for (size_t i = 0; i < input_shape.dims(); ++i) {
      if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) ==
          axes.end()) {
        new_dims.push_back(input_shape[i]);
      }
    }
    if (new_dims.empty()) {
      new_dims.push_back(1);
    }
  }

  return Shape(new_dims);
}

template <typename T>
void ReduceSumLayer::compute(const Tensor& input, const Shape& output_shape,
                             const std::vector<int64_t>& axes,
                             Tensor& output) const {
  const auto& input_data = *input.as<T>();
  std::vector<T> output_data(output_shape.count(), 0);

  const auto& input_shape = input.get_shape();
  const size_t input_rank = input_shape.dims();

  std::vector<bool> is_reduced(input_rank, false);
  for (int64_t axis : axes) {
    is_reduced[axis] = true;
  }

  std::vector<size_t> input_strides(input_rank, 1);
  for (size_t i = input_rank - 1; i > 0; --i) {
    input_strides[i - 1] = input_strides[i] * input_shape[i];
  }

  for (size_t out_idx = 0; out_idx < output_data.size(); ++out_idx) {
    size_t remaining = out_idx;
    size_t input_idx = 0;

    for (size_t out_dim = 0; out_dim < output_shape.dims(); ++out_dim) {
      size_t out_coord = remaining % output_shape[out_dim];
      remaining /= output_shape[out_dim];

      size_t input_dim = 0;
      for (size_t i = 0; i < input_rank; ++i) {
        if (!is_reduced[i]) {
          if (input_dim == out_dim) {
            input_idx += out_coord * input_strides[i];
            break;
          }
          input_dim++;
        }
      }
    }

    T sum = 0;
    size_t reduced_count = 1;
    for (int64_t axis : axes) {
      reduced_count *= input_shape[axis];
    }

    for (size_t offset = 0; offset < reduced_count; ++offset) {
      size_t current_idx = input_idx;
      size_t temp = offset;

      for (int64_t axis : axes) {
        size_t axis_size = input_shape[axis];
        size_t coord = temp % axis_size;
        temp /= axis_size;
        current_idx += coord * input_strides[axis];
      }

      sum += input_data[current_idx];
    }

    output_data[out_idx] = sum;
  }

  output = make_tensor(output_data, output_shape);
}

template void ReduceSumLayer::compute<float>(const Tensor&, const Shape&,
                                             const std::vector<int64_t>&,
                                             Tensor&) const;
template void ReduceSumLayer::compute<int>(const Tensor&, const Shape&,
                                           const std::vector<int64_t>&,
                                           Tensor&) const;

void ReduceSumLayer::run(const Tensor& input, Tensor& output) {
  run(input, Tensor(), output);
}

void ReduceSumLayer::run(const Tensor& input, const Tensor& axes,
                         Tensor& output) {
  if (input.get_shape().count() == 0) {
    output = make_tensor<float>({0.0f}, {});
    return;
  }

  std::vector<int64_t> axes_indices;
  if (axes.get_shape().dims() > 0) {
    if (axes.get_type() == Type::kInt) {
      auto axes_data = axes.as<int>();
      axes_indices.assign(axes_data->begin(), axes_data->end());
    } else {
      throw std::runtime_error("ReduceSum: Axes tensor must be of type int");
    }
  }

  normalize_axes(input.get_shape(), axes_indices);
  Shape output_shape = calculate_output_shape(input.get_shape(), axes_indices);

  switch (input.get_type()) {
    case Type::kFloat:
      compute<float>(input, output_shape, axes_indices, output);
      break;
    case Type::kInt:
      compute<int>(input, output_shape, axes_indices, output);
      break;
    default:
      throw std::runtime_error(
          "ReduceSum: Unsupported input tensor type. Only float and int are "
          "supported");
  }
}

}  // namespace itlab_2023