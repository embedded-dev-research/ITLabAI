#include "layers_oneDNN/ReduceLayer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace it_lab_ai {

void ReduceLayerOneDnn::run(const std::vector<Tensor>& input,
                            std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& in = input[0];
  Type type = in.get_type();
  const Shape& input_shape = in.get_shape();

  normalized_axes_ = axes_;
  normalize_axes(input_shape, normalized_axes_);

  bool need_reinit = !initialized_ || last_type_ != type ||
                     last_input_shape_ != input_shape ||
                     last_axes_ != normalized_axes_;

  if (need_reinit) {
    initialize_onednn(in);
    last_input_shape_ = input_shape;
    last_type_ = type;
    last_axes_ = normalized_axes_;
    initialized_ = true;
  }

  output.resize(1);

  Shape one_dnn_output_shape =
      calculate_output_shape(input_shape, normalized_axes_);

  Shape final_output_shape;
  if (keepdims_) {
    final_output_shape = one_dnn_output_shape;
  } else {
    for (size_t i = 0; i < one_dnn_output_shape.dims(); ++i) {
      if (std::find(normalized_axes_.begin(), normalized_axes_.end(),
                    static_cast<int64_t>(i)) == normalized_axes_.end()) {
        final_output_shape.push_back(one_dnn_output_shape[i]);
      }
    }

    if (final_output_shape.dims() == 0) {
      final_output_shape.push_back(1);
    }
  }

  if (type == Type::kFloat) {
    const auto& src_data = *in.as<float>();
    std::vector<float> src_copy(src_data.begin(), src_data.end());
    std::vector<float> dst_data(one_dnn_output_shape.count());

    dnnl::memory src_mem(src_md_, *engine_, src_copy.data());
    dnnl::memory dst_mem(dst_md_, *engine_, dst_data.data());

    reduction_prim_->execute(
        *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    stream_->wait();

    if (op_ == ReduceLayer::Operation::kMean) {
      size_t reduction_size = 1;
      for (int64_t axis : normalized_axes_) {
        reduction_size *= input_shape[axis];
      }

      float scale = 1.0F / static_cast<float>(reduction_size);
      for (float& v : dst_data) {
        v *= scale;
      }
    }

    std::vector<float> final_data =
        keepdims_ ? dst_data
                  : remove_unit_dims(dst_data, one_dnn_output_shape,
                                     final_output_shape);

    output[0] = make_tensor(final_data, final_output_shape);

  } else if (type == Type::kInt) {
    const auto& src_data = *in.as<int>();
    std::vector<int> src_copy(src_data.begin(), src_data.end());
    std::vector<int> dst_data(one_dnn_output_shape.count());

    dnnl::memory src_mem(src_md_, *engine_, src_copy.data());
    dnnl::memory dst_mem(dst_md_, *engine_, dst_data.data());

    reduction_prim_->execute(
        *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    stream_->wait();

    if (op_ == ReduceLayer::Operation::kMean) {
      size_t reduction_size = 1;
      for (int64_t axis : normalized_axes_) {
        reduction_size *= input_shape[axis];
      }

      for (int& v : dst_data) {
        v /= static_cast<int>(reduction_size);
      }
    }

    if (op_ == ReduceLayer::Operation::kMult) {
      throw std::runtime_error(
          "ReduceLayerOneDnn: kMult is not supported for int type");
    }

    std::vector<int> final_data =
        keepdims_ ? dst_data
                  : remove_unit_dims(dst_data, one_dnn_output_shape,
                                     final_output_shape);

    output[0] = make_tensor(final_data, final_output_shape);
  }
}

template <typename T>
std::vector<T> ReduceLayerOneDnn::remove_unit_dims(
    const std::vector<T>& src_data, const Shape& src_shape,
    const Shape& dst_shape) {
  if (src_shape == dst_shape) {
    return src_data;
  }

  std::vector<T> dst_data(dst_shape.count());
  size_t dst_idx = 0;

  std::vector<size_t> coords(src_shape.dims(), 0);
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    bool keep = true;
    for (size_t dim = 0; dim < coords.size(); ++dim) {
      if (src_shape[dim] == 1 && coords[dim] != 0) {
        keep = false;
        break;
      }
    }

    if (keep) {
      dst_data[dst_idx++] = src_data[src_idx];
    }

    for (size_t dim = coords.size(); dim-- > 0;) {
      ++coords[dim];
      if (coords[dim] < src_shape[dim]) {
        break;
      }
      coords[dim] = 0;
    }
  }

  return dst_data;
}

void ReduceLayerOneDnn::validate_input(const std::vector<Tensor>& input) {
  if (input.size() != 1) {
    throw std::runtime_error(
        "ReduceLayerOneDnn: Expected exactly 1 input tensor");
  }

  const auto& shape = input[0].get_shape();
  if (shape.dims() == 0) {
    throw std::runtime_error("ReduceLayerOneDnn: Scalar input not supported");
  }
}

Shape ReduceLayerOneDnn::calculate_output_shape(
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
    new_dims.resize(input_shape.dims(), 1);
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.dims()); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
      if (!is_axis) {
        new_dims[i] = input_shape[i];
      } else {
        new_dims[i] = 1;
      }
    }
  }

  return Shape(new_dims);
}

void ReduceLayerOneDnn::normalize_axes(const Shape& input_shape,
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

void ReduceLayerOneDnn::initialize_onednn(const Tensor& input) {
  engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  stream_ = std::make_unique<dnnl::stream>(*engine_);

  const Shape& input_shape = input.get_shape();
  std::vector<int64_t> normalized_axes = axes_;
  normalize_axes(input_shape, normalized_axes);

  output_shape_ = calculate_output_shape(input_shape, normalized_axes);

  auto dnnl_type = get_dnnl_data_type(input.get_type());

  auto src_dims = shape_to_dims(input_shape);
  auto dst_dims = shape_to_dims(output_shape_);

  try {
    dnnl::memory::format_tag src_format = pick_format(input_shape.dims());
    dnnl::memory::format_tag dst_format = pick_format(output_shape_.dims());

    src_md_ = dnnl::memory::desc(src_dims, dnnl_type, src_format);
    dst_md_ = dnnl::memory::desc(dst_dims, dnnl_type, dst_format);

    auto algorithm = get_dnnl_algorithm(op_);
    float p = 0.0F;
    float eps = 0.0F;

    auto reduction_pd = dnnl::reduction::primitive_desc(
        *engine_, algorithm, src_md_, dst_md_, p, eps);

    reduction_prim_ = std::make_unique<dnnl::reduction>(reduction_pd);
    src_md_ = reduction_pd.src_desc();
    dst_md_ = reduction_pd.dst_desc();

  } catch (const dnnl::error& e) {
    std::cerr << "Error creating reduction primitive: " << e.what() << '\n';
    std::cerr << "Input dims: ";
    for (auto d : src_dims) {
      std::cerr << d << " ";
    }
    std::cerr << "\nOutput dims: ";
    for (auto d : dst_dims) {
      std::cerr << d << " ";
    }
    std::cerr << "\nOperation: " << static_cast<int>(op_) << '\n';

    throw std::runtime_error("Failed to create oneDNN reduction primitive: " +
                             std::string(e.what()));
  }

  last_input_shape_ = input_shape;
  last_type_ = input.get_type();
  initialized_ = true;
}

dnnl::memory::data_type ReduceLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::algorithm ReduceLayerOneDnn::get_dnnl_algorithm(
    ReduceLayer::Operation op) {
  switch (op) {
    case ReduceLayer::Operation::kSum:
    case ReduceLayer::Operation::kMean:
    case ReduceLayer::Operation::kMult:
      return dnnl::algorithm::reduction_sum;
    case ReduceLayer::Operation::kMax:
      return dnnl::algorithm::reduction_max;
    case ReduceLayer::Operation::kMin:
      return dnnl::algorithm::reduction_min;
    default:
      throw std::invalid_argument("Unsupported reduction operation for oneDNN");
  }
}

std::vector<dnnl::memory::dim> ReduceLayerOneDnn::shape_to_dims(
    const Shape& shape) {
  std::vector<dnnl::memory::dim> dims;
  for (size_t i = 0; i < shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
  }

  if (dims.empty()) {
    dims.push_back(1);
  }

  return dims;
}

dnnl::memory::format_tag ReduceLayerOneDnn::pick_format(size_t ndims) {
  switch (ndims) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    default:
      if (ndims == 6) {
        return dnnl::memory::format_tag::abcdef;
      }
      throw std::runtime_error("Unsupported tensor dimensionality: " +
                               std::to_string(ndims));
  }
}

}  // namespace it_lab_ai
