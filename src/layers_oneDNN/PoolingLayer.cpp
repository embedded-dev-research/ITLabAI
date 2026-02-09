#include "layers_oneDNN/PoolingLayer.hpp"

#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void PoolingLayerOneDnn::run(const std::vector<Tensor>& input,
                             std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& in = input[0];
  Type type = in.get_type();

  bool need_reinit =
      !initialized_ || last_type_ != type || last_shape_ != in.get_shape();

  if (need_reinit) {
    initialize_onednn(in.get_shape(), type);
  }
  output.resize(1);

  if (type == Type::kFloat) {
    const auto& src = *in.as<float>();
    std::vector<float> dst(output_shape_.count());

    dnnl::memory src_mem(src_memory_desc_, *engine_,
                         const_cast<float*>(src.data()));
    dnnl::memory dst_mem(dst_memory_desc_, *engine_, dst.data());

    pool_prim_->execute(*stream_,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst, output_shape_);
  } else if (type == Type::kInt) {
    const auto& src = *in.as<int>();
    std::vector<int> dst(output_shape_.count());

    dnnl::memory src_mem(src_memory_desc_, *engine_,
                         const_cast<int*>(src.data()));
    dnnl::memory dst_mem(dst_memory_desc_, *engine_, dst.data());

    pool_prim_->execute(*stream_,
                        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst, output_shape_);
  }
}

void PoolingLayerOneDnn::validate_input(const std::vector<Tensor>& input) {
  if (input.size() != 1) {
    throw std::runtime_error(
        "PoolingLayerOneDnn: Expected exactly 1 input tensor");
  }

  const auto& shape = input[0].get_shape();
  if (shape.dims() < 2) {
    throw std::runtime_error(
        "PoolingLayerOneDnn: Input must have at least 2 dimensions");
  }
}

Shape PoolingLayerOneDnn::calculate_output_shape(
    const Shape& input_shape) const {
  Shape output_shape = input_shape;

  if (poolingShape_[0] == 0 && poolingShape_[1] == 0) {
    for (size_t i = 0; i < std::min(static_cast<size_t>(2), input_shape.dims());
         ++i) {
      output_shape[i] = input_shape[i];
    }
    for (size_t i = 2; i < input_shape.dims(); ++i) {
      output_shape[i] = 1;
    }
    return output_shape;
  }

  size_t spatial_dims = poolingShape_.dims();

  for (size_t i = 0; i < spatial_dims; ++i) {
    size_t input_idx = input_shape.dims() - spatial_dims + i;
    size_t input_size = input_shape[input_idx];
    size_t kernel_size = poolingShape_[i];
    size_t stride = strides_[i];

    size_t pad_front = pads_[i];
    size_t pad_back = pads_[i + spatial_dims];
    size_t dilation = dilations_[i];

    size_t effective_kernel_size = (kernel_size - 1) * dilation + 1;

    size_t output_size;
    if (ceil_mode_) {
      output_size = static_cast<size_t>(std::ceil(
                        static_cast<float>(input_size + pad_front + pad_back -
                                           effective_kernel_size) /
                        static_cast<float>(stride))) +
                    1;
    } else {
      output_size = static_cast<size_t>(std::floor(
                        static_cast<float>(input_size + pad_front + pad_back -
                                           effective_kernel_size) /
                        static_cast<float>(stride))) +
                    1;
    }

    output_shape[input_idx] = output_size;
  }

  return output_shape;
}

void PoolingLayerOneDnn::initialize_onednn(const Shape& shape, Type data_type) {
  output_shape_ = calculate_output_shape(shape);

  std::vector<dnnl::memory::dim> src_dims;
  std::vector<dnnl::memory::dim> dst_dims;

  if (shape.dims() == 4) {
    for (size_t i = 0; i < 4; ++i) {
      src_dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
    }
    for (size_t i = 0; i < 4; ++i) {
      dst_dims.push_back(static_cast<dnnl::memory::dim>(output_shape_.at(i)));
    }
  } else if (shape.dims() == 3) {
    src_dims.push_back(1);
    dst_dims.push_back(1);
    for (size_t i = 0; i < 3; ++i) {
      src_dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
      dst_dims.push_back(static_cast<dnnl::memory::dim>(output_shape_.at(i)));
    }
  } else if (shape.dims() == 2) {
    src_dims = {1, 1, static_cast<dnnl::memory::dim>(shape[0]),
                static_cast<dnnl::memory::dim>(shape[1])};
    dst_dims = {1, 1, static_cast<dnnl::memory::dim>(output_shape_[0]),
                static_cast<dnnl::memory::dim>(output_shape_[1])};
  } else {
    throw std::runtime_error("Unsupported shape dimensions for pooling: " +
                             std::to_string(shape.dims()));
  }

  auto dnnl_type = get_dnnl_data_type(data_type);

  src_memory_desc_ =
      dnnl::memory::desc(src_dims, dnnl_type, dnnl::memory::format_tag::nchw);
  dst_memory_desc_ =
      dnnl::memory::desc(dst_dims, dnnl_type, dnnl::memory::format_tag::nchw);

  dnnl::memory::dims strides = {static_cast<dnnl::memory::dim>(strides_[0]),
                                static_cast<dnnl::memory::dim>(strides_[1])};

  dnnl::memory::dims kernel;
  bool is_global_pool = (poolingShape_[0] == 0 && poolingShape_[1] == 0);

  if (is_global_pool) {
    kernel = {static_cast<dnnl::memory::dim>(src_dims[2]),
              static_cast<dnnl::memory::dim>(src_dims[3])};
    strides = {1, 1};
  } else {
    kernel = {static_cast<dnnl::memory::dim>(poolingShape_[0]),
              static_cast<dnnl::memory::dim>(poolingShape_[1])};
  }

  dnnl::memory::dims dilations = {
      static_cast<dnnl::memory::dim>(dilations_[0] - 1),
      static_cast<dnnl::memory::dim>(dilations_[1] - 1)};

  dnnl::memory::dims padding_l = {static_cast<dnnl::memory::dim>(pads_[0]),
                                  static_cast<dnnl::memory::dim>(pads_[2])};

  dnnl::memory::dims padding_r = {static_cast<dnnl::memory::dim>(pads_[1]),
                                  static_cast<dnnl::memory::dim>(pads_[3])};

  if (ceil_mode_ && !is_global_pool) {
    for (size_t i = 0; i < 2; ++i) {
      auto input_size = static_cast<size_t>(src_dims[2 + i]);
      auto kernel_size = static_cast<size_t>(kernel[i]);
      auto stride = static_cast<size_t>(strides[i]);
      size_t dilation = static_cast<size_t>(dilations[i]) + 1;
      auto pad_front = static_cast<size_t>(padding_l[i]);
      auto pad_back = static_cast<size_t>(padding_r[i]);
      size_t effective_kernel = (kernel_size - 1) * dilation + 1;
      auto output_size = static_cast<size_t>(dst_dims[2 + i]);
      size_t needed_pad_back = (output_size - 1) * stride + effective_kernel -
                               input_size - pad_front;

      if (needed_pad_back > pad_back) {
        padding_r[i] = static_cast<dnnl::memory::dim>(needed_pad_back);
      }
    }
  }

  try {
    dnnl::pooling_forward::primitive_desc pool_pd(
        *engine_, dnnl::prop_kind::forward_inference, get_PoolType(),
        src_memory_desc_, dst_memory_desc_, strides, kernel, dilations,
        padding_l, padding_r);

    pool_prim_ = std::make_unique<dnnl::pooling_forward>(pool_pd);
  } catch (const dnnl::error& e) {
    std::cerr << "Error creating pooling primitive: " << e.what() << '\n';
    throw std::runtime_error("Failed to create pooling primitive: " +
                             std::string(e.what()));
  }

  last_shape_ = shape;
  last_type_ = data_type;
  initialized_ = true;
}

dnnl::memory::data_type PoolingLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::algorithm PoolingLayerOneDnn::get_PoolType() const {
  if (poolingType_ == "average" || poolingType_ == "Average") {
    return dnnl::algorithm::pooling_avg_include_padding;
  }
  if (poolingType_ == "max" || poolingType_ == "Max") {
    return dnnl::algorithm::pooling_max;
  }

  throw std::invalid_argument("Unsupported pooling type for oneDNN: " +
                              poolingType_);
}

}  // namespace it_lab_ai