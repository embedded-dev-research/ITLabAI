#include "layers_oneDNN/EWLayer.hpp"

#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void EwLayerOneDnn::run(const std::vector<Tensor>& input,
                        std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& input_tensor = input[0];
  Type data_type = input_tensor.get_type();

  if (!initialized_) {
    initialize_onednn(input_tensor.get_shape(), data_type);
  }

  try {
    if (data_type == Type::kFloat) {
      const std::vector<float>& input_data = *input_tensor.as<float>();
      std::vector<float> output_data(input_data.size());
      dnnl::memory src_mem = dnnl::memory(
          memory_desc_, *engine_, const_cast<float*>(input_data.data()));
      dnnl::memory dst_mem =
          dnnl::memory(memory_desc_, *engine_, output_data.data());
      eltwise_prim_->execute(
          *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      stream_->wait();
      output[0] = make_tensor(output_data, input_tensor.get_shape());
    } else if (data_type == Type::kInt) {
      const std::vector<int>& input_data = *input_tensor.as<int>();
      std::vector<int> output_data(input_data.size());

      std::vector<float> float_input;
      float_input.reserve(input_data.size());
      for (int val : input_data) {
        float_input.push_back(static_cast<float>(val));
      }

      std::vector<float> float_output(input_data.size());

      dnnl::memory src_mem =
          dnnl::memory(memory_desc_, *engine_, float_input.data());
      dnnl::memory dst_mem =
          dnnl::memory(memory_desc_, *engine_, float_output.data());
      eltwise_prim_->execute(
          *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
      stream_->wait();

      for (size_t i = 0; i < float_output.size(); ++i) {
        output_data[i] = static_cast<int>(std::round(float_output[i]));
      }
      output[0] = make_tensor(output_data, input_tensor.get_shape());
    } else {
      throw std::runtime_error("EwLayerOneDnn: Unsupported data type");
    }

  } catch (const std::exception& e) {
    std::cerr << "oneDNN execution failed: " << e.what() << std::endl;
    throw;
  }
}

void EwLayerOneDnn::validate_input(const std::vector<Tensor>& input) const {
  if (input.size() != 1) {
    throw std::runtime_error("EwLayerOneDnn: Expected exactly 1 input tensor");
  }

  if (!is_function_supported(func_)) {
    throw std::invalid_argument("Unsupported function for oneDNN: " + func_);
  }

  Type data_type = input[0].get_type();
  if (data_type != Type::kFloat && data_type != Type::kInt) {
    throw std::runtime_error(
        "EwLayerOneDnn supports only float and int tensors");
  }
}

void EwLayerOneDnn::initialize_onednn(const Shape& shape, Type data_type) {
  try {
    engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    stream_ = std::make_unique<dnnl::stream>(*engine_);

    std::vector<dnnl::memory::dim> dims;
    for (size_t i = 0; i < shape.dims(); i++) {
      dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
    }

    dnnl::memory::format_tag format;
    switch (dims.size()) {
      case 1:
        format = dnnl::memory::format_tag::a;
        break;
      case 2:
        format = dnnl::memory::format_tag::ab;
        break;
      case 3:
        format = dnnl::memory::format_tag::abc;
        break;
      case 4:
        format = dnnl::memory::format_tag::abcd;
        break;
      case 5:
        format = dnnl::memory::format_tag::abcde;
        break;
      default:
        throw std::invalid_argument("Unsupported tensor dimensionality: " +
                                    std::to_string(dims.size()));
    }

    dnnl::memory::data_type dnnl_data_type;
    if (data_type == Type::kFloat) {
      dnnl_data_type = dnnl::memory::data_type::f32;
    } else {
      dnnl_data_type = dnnl::memory::data_type::f32;
    }

    memory_desc_ = dnnl::memory::desc(dims, dnnl_data_type, format);

    dnnl::algorithm algo = get_algorithm();

    float primitive_alpha = 0.0F;
    float primitive_beta = 0.0F;

    if (func_ == "relu") {
      primitive_alpha = 0.0F;
    } else if (func_ == "linear") {
      primitive_alpha = alpha_;
      primitive_beta = beta_;
    }

    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(
        *engine_, dnnl::prop_kind::forward_inference, algo, memory_desc_,
        memory_desc_, primitive_alpha, primitive_beta);

    eltwise_prim_ = std::make_unique<dnnl::eltwise_forward>(eltwise_pd);

    initialized_ = true;

  } catch (const std::exception& e) {
    std::cerr << "oneDNN initialization failed for function '" << func_
              << "': " << e.what() << std::endl;
    throw;
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

}  // namespace it_lab_ai