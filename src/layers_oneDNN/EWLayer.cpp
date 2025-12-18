#include "layers_oneDNN/EWLayer.hpp"

#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void EwLayerOneDnn::run(const std::vector<Tensor>& input,
                        std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& in = input[0];
  Type type = in.get_type();

  bool need_reinit =
      !initialized_ || last_type_ != type || last_shape_ != in.get_shape();

  if (need_reinit) {
    initialize_onednn(in.get_shape(), type);
  }

  if (type == Type::kFloat) {
    const auto& src = *in.as<float>();
    std::vector<float> dst(src.size());

    dnnl::memory src_mem(memory_desc_, *engine_,
                         const_cast<float*>(src.data()));
    dnnl::memory dst_mem(memory_desc_, *engine_, dst.data());

    eltwise_prim_->execute(*stream_,
                           {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst, in.get_shape());
  } else if (type == Type::kInt) {
    const auto& src = *in.as<int>();
    std::vector<int> dst(src.size());

    dnnl::memory src_mem(memory_desc_, *engine_, const_cast<int*>(src.data()));
    dnnl::memory dst_mem(memory_desc_, *engine_, dst.data());

    eltwise_prim_->execute(*stream_,
                           {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst, in.get_shape());
  }
}

void EwLayerOneDnn::validate_input(const std::vector<Tensor>& input) const {
  if (input.size() != 1) {
    throw std::runtime_error("EwLayerOneDnn: Expected exactly 1 input tensor");
  }

  if (!is_function_supported(func_)) {
    throw std::invalid_argument("Unsupported function for oneDNN: " + func_);
  }
}

void EwLayerOneDnn::initialize_onednn(const Shape& shape, Type data_type) {
  engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  stream_ = std::make_unique<dnnl::stream>(*engine_);

  std::vector<dnnl::memory::dim> dims;
  for (size_t i = 0; i < shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
  }

  auto format = pick_format(dims.size());
  auto dnnl_type = get_dnnl_data_type(data_type);

  memory_desc_ = dnnl::memory::desc(dims, dnnl_type, format);

  float alpha = 0.0F;
  float beta = 0.0F;

  if (func_ == "linear") {
    alpha = alpha_;
    beta = beta_;
  }

  auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(
      *engine_, dnnl::prop_kind::forward_inference, get_algorithm(),
      memory_desc_, memory_desc_, alpha, beta);

  eltwise_prim_ = std::make_unique<dnnl::eltwise_forward>(eltwise_pd);

  last_shape_ = shape;
  last_type_ = data_type;
  initialized_ = true;
}

dnnl::memory::data_type EwLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::algorithm EwLayerOneDnn::get_algorithm() const {
  if (func_ == "relu") {
    return dnnl::algorithm::eltwise_relu;
  }
  if (func_ == "tanh") {
    return dnnl::algorithm::eltwise_tanh;
  }
  if (func_ == "sigmoid") {
    return dnnl::algorithm::eltwise_logistic;
  }
  if (func_ == "linear") {
    return dnnl::algorithm::eltwise_linear;
  }

  throw std::invalid_argument("Unsupported function for oneDNN: " + func_);
}

bool EwLayerOneDnn::is_function_supported(const std::string& function) {
  return (function == "relu" || function == "tanh" || function == "sigmoid" ||
          function == "linear");
}

dnnl::memory::format_tag EwLayerOneDnn::pick_format(size_t ndims) {
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
      throw std::invalid_argument("Unsupported tensor dimensionality: " +
                                  std::to_string(ndims));
  }
}

}  // namespace it_lab_ai
