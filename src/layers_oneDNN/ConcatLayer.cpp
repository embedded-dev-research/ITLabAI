#include "layers_oneDNN/ConcatLayer.hpp"

#include <stdexcept>

namespace it_lab_ai {

void ConcatLayerOneDnn::run(const std::vector<Tensor>& input,
                            std::vector<Tensor>& output) {
  validate_input(input);

  if (input.size() == 1) {
    output = input;
    return;
  }

  Type type = input[0].get_type();

  bool need_reinit = !initialized_ || last_type_ != type ||
                     last_shapes_.size() != input.size();

  if (!need_reinit) {
    for (size_t i = 0; i < input.size(); ++i) {
      if (last_shapes_[i] != input[i].get_shape()) {
        need_reinit = true;
        break;
      }
    }
  }

  if (need_reinit) {
    initialize_onednn(input);
  }

  output.resize(1);

  if (type == Type::kFloat) {
    for (size_t i = 0; i < input.size(); ++i) {
      if (last_type_ == Type::kFloat) {
        src_mems_[i].set_data_handle(
            const_cast<float*>(input[i].as<float>()->data()));
      } else {
        src_mems_[i].set_data_handle(
            const_cast<int*>(input[i].as<int>()->data()));
      }

      args_[DNNL_ARG_MULTIPLE_SRC + i] = src_mems_[i];
    }

    args_[DNNL_ARG_DST] = dst_mem_;

    concat_prim_->execute(*stream_, args_);
    stream_->wait();

    output[0] = make_tensor(dst_buffer_f32_, output_shape_);
  } else if (type == Type::kInt) {
    for (size_t i = 0; i < input.size(); ++i) {
      src_mems_[i].set_data_handle(
          const_cast<int*>(input[i].as<int>()->data()));
      args_[DNNL_ARG_MULTIPLE_SRC + i] = src_mems_[i];
    }

    args_[DNNL_ARG_DST] = dst_mem_;

    concat_prim_->execute(*stream_, args_);
    stream_->wait();

    output[0] = make_tensor(dst_buffer_s32_, output_shape_);
  }
}

void ConcatLayerOneDnn::validate_input(const std::vector<Tensor>& input) {
  Type type = input[0].get_type();
  const Shape& base = input[0].get_shape();

  for (size_t i = 1; i < input.size(); ++i) {
    if (input[i].get_type() != type) {
      throw std::runtime_error(
          "ConcatLayerOneDnn: All tensors must have same type");
    }

    if (input[i].get_shape().dims() != base.dims()) {
      throw std::runtime_error(
          "ConcatLayerOneDnn: All tensors must have same rank");
    }
  }
}

void ConcatLayerOneDnn::initialize_onednn(const std::vector<Tensor>& input) {
  if (!engine_) {
    engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  }
  if (!stream_) {
    stream_ = std::make_unique<dnnl::stream>(*engine_);
  }

  size_t rank = input[0].get_shape().dims();
  int64_t axis = normalize_axis(axis_, rank);

  last_type_ = input[0].get_type();
  auto type = get_dnnl_data_type(last_type_);

  auto layout = pick_format(rank);

  src_mds_.clear();
  for (const auto& t : input) {
    src_mds_.emplace_back(shape_to_dims(t.get_shape()), type, layout);
  }

  output_shape_ = calculate_output_shape(input, axis);

  dst_md_ = dnnl::memory::desc(shape_to_dims(output_shape_), type, layout);

  auto concat_pd =
      dnnl::concat::primitive_desc(*engine_, dst_md_, axis, src_mds_);
  concat_prim_ = std::make_unique<dnnl::concat>(concat_pd);

  dst_md_ = concat_pd.dst_desc();
  src_mds_.clear();
  for (size_t i = 0; i < input.size(); ++i) {
    src_mds_.push_back(concat_pd.src_desc(i));
  }

  size_t n = input.size();
  src_mems_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    src_mems_[i] = dnnl::memory(src_mds_[i], *engine_, nullptr);
  }

  size_t out_size = output_shape_.count();
  if (last_type_ == Type::kFloat) {
    dst_buffer_f32_.resize(out_size);
    dst_mem_ = dnnl::memory(dst_md_, *engine_, dst_buffer_f32_.data());
  } else {
    dst_buffer_s32_.resize(out_size);
    dst_mem_ = dnnl::memory(dst_md_, *engine_, dst_buffer_s32_.data());
  }

  args_.clear();
  for (size_t i = 0; i < n; ++i) {
    args_[DNNL_ARG_MULTIPLE_SRC + i] = src_mems_[i];
  }
  args_[DNNL_ARG_DST] = dst_mem_;

  last_shapes_.clear();
  for (const auto& t : input) {
    last_shapes_.push_back(t.get_shape());
  }

  initialized_ = true;
}

dnnl::memory::data_type ConcatLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::memory::format_tag ConcatLayerOneDnn::pick_format(size_t ndims) {
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
      return dnnl::memory::format_tag::any;
  }
}

std::vector<dnnl::memory::dim> ConcatLayerOneDnn::shape_to_dims(
    const Shape& shape) {
  std::vector<dnnl::memory::dim> dims;

  for (size_t i = 0; i < shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
  }

  return dims;
}

Shape ConcatLayerOneDnn::calculate_output_shape(
    const std::vector<Tensor>& inputs, int64_t axis) {
  const Shape& base = inputs[0].get_shape();

  std::vector<size_t> dims(base.dims());

  for (size_t i = 0; i < base.dims(); ++i) {
    dims[i] = base[i];
  }

  dims[axis] = 0;

  for (const auto& t : inputs) {
    dims[axis] += t.get_shape()[axis];
  }

  return Shape(dims);
}

int64_t ConcatLayerOneDnn::normalize_axis(int64_t axis, size_t rank) {
  if (axis < 0) {
    axis += rank;
  }

  if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
    throw std::runtime_error("ConcatLayerOneDnn: axis out of range");
  }

  return axis;
}

}  // namespace it_lab_ai
