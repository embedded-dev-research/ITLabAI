#include "layers/ReduceLayer.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace it_lab_ai {

ReduceLayer::ReduceLayer(Operation op, int64_t keepdims, const Tensor& axes)
    : Layer(kReduce), op_(op), keepdims_(keepdims), axes_(axes) {}

void ReduceLayer::normalize_axes(const Shape& input_shape,
                                 std::vector<int64_t>& axes) {
  const auto rank = static_cast<int64_t>(input_shape.dims());

  if (rank == 0) {
    if (!axes.empty()) {
      throw std::runtime_error("ReduceLayer: Axis specified for scalar input");
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
      throw std::runtime_error(
          "ReduceLayer: Axis out of range. Valid range is [-" +
          std::to_string(rank) + ", " + std::to_string(rank - 1) + "]");
    }

    if (axis < 0) {
      axis += rank;
    }
  }

  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
}

Shape ReduceLayer::calculate_output_shape(
    const Shape& input_shape, const std::vector<int64_t>& axes) const {
  if (input_shape.dims() == 0) {
    return Shape({});
  }

  std::vector<size_t> new_dims;

  if (keepdims_) {
    new_dims.resize(input_shape.dims(), 1);
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.dims()); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
      if (!is_axis) {
        new_dims[i] = input_shape[i];
      }
    }
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.dims()); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
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
void ReduceLayer::compute(const Tensor& input, const Shape& output_shape,
                          const std::vector<int64_t>& axes,
                          Tensor& output) const {
  const auto& input_data = *input.as<T>();
  std::vector<T> output_data(output_shape.count());
  std::vector<size_t> counts(output_shape.count(), 0);

  switch (op_) {
    case Operation::kSum:
    case Operation::kMean:
      std::fill(output_data.begin(), output_data.end(), T(0));
      break;
    case Operation::kMult:
      std::fill(output_data.begin(), output_data.end(), T(1));
      break;
    case Operation::kMax:
      std::fill(output_data.begin(), output_data.end(),
                std::numeric_limits<T>::lowest());
      break;
    case Operation::kMin:
      std::fill(output_data.begin(), output_data.end(),
                std::numeric_limits<T>::max());
      break;
  }

  const auto& input_shape = input.get_shape();
  const auto input_rank = static_cast<int64_t>(input_shape.dims());

  std::vector<size_t> in_coords(input_rank, 0);
  for (size_t in_idx = 0; in_idx < input_data.size(); ++in_idx) {
    std::vector<size_t> out_coords;
    if (keepdims_) {
      out_coords.resize(input_rank, 0);
      for (int64_t i = 0; i < input_rank; ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
          out_coords[i] = in_coords[i];
        }
      }
    } else {
      for (int64_t i = 0; i < input_rank; ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
          out_coords.push_back(in_coords[i]);
        }
      }
    }

    size_t out_idx = 0;
    size_t stride = 1;
    for (size_t i = out_coords.size(); i-- > 0;) {
      out_idx += out_coords[i] * stride;
      stride *= output_shape[i];
    }

    switch (op_) {
      case Operation::kSum:
      case Operation::kMean:
        output_data[out_idx] += input_data[in_idx];
        counts[out_idx]++;
        break;
      case Operation::kMult:
        output_data[out_idx] *= input_data[in_idx];
        break;
      case Operation::kMax:
        if (input_data[in_idx] > output_data[out_idx]) {
          output_data[out_idx] = input_data[in_idx];
        }
        break;
      case Operation::kMin:
        if (input_data[in_idx] < output_data[out_idx]) {
          output_data[out_idx] = input_data[in_idx];
        }
        break;
    }

    for (int64_t i = input_rank; i-- > 0;) {
      ++in_coords[i];
      if (in_coords[i] < input_shape[i]) break;
      in_coords[i] = 0;
    }
  }

  if (op_ == Operation::kMean) {
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (counts[i] != 0) {
        output_data[i] /= static_cast<T>(counts[i]);
      }
    }
  }

  output = make_tensor(output_data, output_shape);
}

template void ReduceLayer::compute<float>(const Tensor&, const Shape&,
                                          const std::vector<int64_t>&,
                                          Tensor&) const;
template void ReduceLayer::compute<int>(const Tensor&, const Shape&,
                                        const std::vector<int64_t>&,
                                        Tensor&) const;

void ReduceLayer::run(const std::vector<Tensor>& input,
                      std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("ReduceLayer: Input tensors not 1");
  }

  if (input[0].get_shape().count() == 0) {
    output[0] = make_tensor<float>({0.0F}, {});
    return;
  }

  std::vector<int64_t> axes_indices;
  if (axes_.get_shape().dims() > 0) {
    if (axes_.get_type() == Type::kInt) {
      const auto* axes_data = axes_.as<int>();
      axes_indices.assign(axes_data->begin(), axes_data->end());
    } else {
      throw std::runtime_error("ReduceLayer: Axes tensor must be of type int");
    }
  }

  normalize_axes(input[0].get_shape(), axes_indices);
  Shape output_shape =
      calculate_output_shape(input[0].get_shape(), axes_indices);

  switch (input[0].get_type()) {
    case Type::kFloat:
      compute<float>(input[0], output_shape, axes_indices, output[0]);
      break;
    case Type::kInt:
      compute<int>(input[0], output_shape, axes_indices, output[0]);
      break;
    default:
      throw std::runtime_error(
          "ReduceLayer: Unsupported input tensor type. Only float and int are "
          "supported");
  }
}

}  // namespace it_lab_ai