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
    std::iota(axes.begin(), axes.end(), 1);
    return;
  }

  for (auto& axis : axes) {
    if (axis < 1 || axis > rank) {
      throw std::runtime_error(
          "ReduceSum: Axis out of range. Use 1-based indexing");
    }
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
    new_dims.resize(input_shape.dims(), 1);
    for (size_t i = 0; i < input_shape.dims(); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(),
                               static_cast<int64_t>(i + 1)) != axes.end();
      if (!is_axis) {
        new_dims[i] = input_shape[i];
      }
    }
  } else {
    for (size_t i = 0; i < input_shape.dims(); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(),
                               static_cast<int64_t>(i + 1)) != axes.end();
      if (!is_axis) {
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

  std::vector<size_t> reduced_axes;
  for (auto axis : axes) {
    reduced_axes.push_back(static_cast<size_t>(axis - 1));
  }

  std::vector<size_t> strides(input_rank, 1);
  for (size_t i = input_rank - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * input_shape[i];
  }

  std::vector<size_t> axis_mapping;
  for (size_t i = 0; i < input_rank; ++i) {
    if (std::find(reduced_axes.begin(), reduced_axes.end(), i) ==
        reduced_axes.end()) {
      axis_mapping.push_back(i);
    }
  }

  std::vector<size_t> out_strides(output_shape.dims(), 1);
  for (size_t i = output_shape.dims() - 1; i > 0; --i) {
    out_strides[i - 1] = out_strides[i] * output_shape[i];
  }

  std::vector<size_t> in_coords(input_rank, 0);
  for (size_t in_idx = 0; in_idx < input_data.size(); ++in_idx) {
    std::vector<size_t> out_coords;
    for (size_t i = 0; i < input_rank; ++i) {
      if (std::find(reduced_axes.begin(), reduced_axes.end(), i) ==
          reduced_axes.end()) {
        out_coords.push_back(in_coords[i]);
      }
    }

    size_t out_idx = 0;
    for (size_t i = 0; i < out_coords.size(); ++i) {
      out_idx += out_coords[i] * out_strides[i];
    }

    if (keepdims_) {
      std::vector<size_t> full_out_coords;
      size_t out_pos = 0;
      for (size_t i = 0; i < input_rank; ++i) {
        if (std::find(reduced_axes.begin(), reduced_axes.end(), i) !=
            reduced_axes.end()) {
          full_out_coords.push_back(0);
        } else {
          full_out_coords.push_back(out_coords[out_pos++]);
        }
      }
      out_idx = 0;
      for (size_t i = 0; i < full_out_coords.size(); ++i) {
        out_idx += full_out_coords[i] * out_strides[i];
      }
    }

    output_data[out_idx] += input_data[in_idx];

    for (size_t i = input_rank - 1;; --i) {
      ++in_coords[i];
      if (in_coords[i] < input_shape[i] || i == 0) {
        break;
      }
      in_coords[i] = 0;
    }
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