#include "layers/ReduceLayer.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>

namespace it_lab_ai {

ReduceLayer::ReduceLayer(Operation op, int64_t keepdims,
                         const std::vector<int64_t>& axes)
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
                          const std::vector<int64_t>& axes, Tensor& output,
                          ParBackend backend) const {
  const auto& input_data = *input.as<T>();
  const auto& input_shape = input.get_shape();
  const auto input_rank = static_cast<int64_t>(input_shape.dims());

  std::vector<T> output_data(output_shape.count());

  parallel::Options options;
  options.backend = backend;

  parallel::parallel_for(output_shape.count(), [&](size_t out_idx) {
    std::vector<size_t> out_coords(output_shape.dims(), 0);
    size_t tmp = out_idx;
    for (size_t i = output_shape.dims(); i-- > 0;) {
      out_coords[i] = tmp % output_shape[i];
      tmp /= output_shape[i];
    }

    T local_result;
    size_t local_count = 0;

    switch (op_) {
      case Operation::kSum:
      case Operation::kMean:
        local_result = T(0);
        break;
      case Operation::kMult:
        local_result = T(1);
        break;
      case Operation::kMax:
        local_result = std::numeric_limits<T>::lowest();
        break;
      case Operation::kMin:
        local_result = std::numeric_limits<T>::max();
        break;
    }

    std::vector<size_t> in_coords(input_rank, 0);

    std::function<void(int64_t)> iterate_inputs = [&](int64_t axis_idx) {
      if (axis_idx == input_rank) {
        size_t in_idx = input_shape.get_index(in_coords);
        const T& val = input_data[in_idx];

        switch (op_) {
          case Operation::kSum:
          case Operation::kMean:
            local_result += val;
            local_count++;
            break;
          case Operation::kMult:
            local_result *= val;
            break;
          case Operation::kMax:
            if (local_count == 0 || val > local_result) {
              local_result = val;
            }
            local_count++;
            break;
          case Operation::kMin:
            if (local_count == 0 || val < local_result) {
              local_result = val;
            }
            local_count++;
            break;
        }
        return;
      }

      bool is_reduce_axis =
          std::find(axes.begin(), axes.end(), axis_idx) != axes.end();

      if (is_reduce_axis) {
        for (size_t coord = 0; coord < input_shape[axis_idx]; ++coord) {
          in_coords[axis_idx] = coord;
          iterate_inputs(axis_idx + 1);
        }
      } else {
        int64_t out_axis = 0;
        for (int64_t i = 0; i < axis_idx; ++i) {
          if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            out_axis++;
          }
        }
        in_coords[axis_idx] =
            keepdims_ ? out_coords[axis_idx] : out_coords[out_axis];
        iterate_inputs(axis_idx + 1);
      }
    };

    iterate_inputs(0);

    if (op_ == Operation::kMean && local_count > 0) {
      output_data[out_idx] = local_result / static_cast<T>(local_count);
    } else {
      output_data[out_idx] = local_result;
    }
  }, options);

  output = make_tensor(output_data, output_shape);
}

void ReduceLayer::run(const std::vector<Tensor>& input,
                      std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void ReduceLayer::run(const std::vector<Tensor>& input,
                      std::vector<Tensor>& output,
                      const RuntimeOptions& options) {
  if (input.size() != 1) {
    throw std::runtime_error("ReduceLayer: Input tensors not 1");
  }

  if (input[0].get_shape().count() == 0) {
    output[0] = make_tensor<float>({0.0F}, {});
    return;
  }

  std::vector<int64_t> axes_indices = axes_;
  normalize_axes(input[0].get_shape(), axes_indices);

  Shape output_shape =
      calculate_output_shape(input[0].get_shape(), axes_indices);

  ParBackend backend = options.par_backend;

  switch (input[0].get_type()) {
    case Type::kFloat:
      compute<float>(input[0], output_shape, axes_indices, output[0], backend);
      break;
    case Type::kInt:
      compute<int>(input[0], output_shape, axes_indices, output[0], backend);
      break;
    default:
      throw std::runtime_error(
          "ReduceLayer: Unsupported input tensor type. Only float and int are "
          "supported");
  }
}

template void ReduceLayer::compute<float>(const Tensor&, const Shape&,
                                          const std::vector<int64_t>&, Tensor&,
                                          ParBackend) const;
template void ReduceLayer::compute<int>(const Tensor&, const Shape&,
                                        const std::vector<int64_t>&, Tensor&,
                                        ParBackend) const;

}  // namespace it_lab_ai
