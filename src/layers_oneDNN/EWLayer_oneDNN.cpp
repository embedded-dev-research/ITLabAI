#include "layers_oneDNN/EWLayer_oneDNN.hpp"

#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void EWLayer_oneDNN::run(const std::vector<Tensor>& input,
                         std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& input_tensor = input[0];

  if (!initialized_) {
    initialize_onednn(input_tensor.get_shape());
  }
  if (input_tensor.get_type() != Type::kFloat) {
    throw std::runtime_error("oneDNN EWLayer supports only float tensors");
  }

  try {
    const std::vector<float>& input_data = *input_tensor.as<float>();
    std::vector<float> output_data(input_data.size());
    dnnl::memory src_mem = dnnl::memory(memory_desc_, *engine_,
                                        const_cast<float*>(input_data.data()));
    dnnl::memory dst_mem =
        dnnl::memory(memory_desc_, *engine_, output_data.data());
    eltwise_prim_->execute(*stream_,
                           {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    stream_->wait();
    output[0] = make_tensor(output_data, input_tensor.get_shape());

  } catch (const std::exception& e) {
    std::cerr << "oneDNN execution failed: " << e.what() << std::endl;
    throw;
  }
}

void EWLayer_oneDNN::validate_input(const std::vector<Tensor>& input) const {
  if (input.size() != 1) {
    throw std::runtime_error("EWLayer_oneDNN: Expected exactly 1 input tensor");
  }

  if (!is_function_supported(func_)) {
    throw std::invalid_argument("Unsupported function for oneDNN: " + func_);
  }
}

void EWLayer_oneDNN::initialize_onednn(const Shape& shape) {
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

    memory_desc_ =
        dnnl::memory::desc(dims, dnnl::memory::data_type::f32, format);

    dnnl::algorithm algo = get_algorithm();

    float primitive_alpha = 0.0f;
    float primitive_beta = 0.0f;

    if (func_ == "relu") {
      primitive_alpha = 0.0f;
    } else if (func_ == "linear") {
      primitive_alpha = alpha_;
      primitive_beta = beta_;
    }

    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(
        *engine_, dnnl::prop_kind::forward_inference, algo, memory_desc_,
        memory_desc_, primitive_alpha, primitive_beta);

    eltwise_prim_ = std::make_unique<dnnl::eltwise_forward>(eltwise_pd);

    initialized_ = true;

    for (size_t i = 0; i < dims.size(); ++i) {
      std::cout << dims[i];
      if (i < dims.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "oneDNN initialization failed for function '" << func_
              << "': " << e.what() << std::endl;
    throw;
  }
}

dnnl::algorithm EWLayer_oneDNN::get_algorithm() const {
  if (func_ == "relu") {
    return dnnl::algorithm::eltwise_relu;
  } else if (func_ == "tanh") {
    return dnnl::algorithm::eltwise_tanh;
  } else if (func_ == "sigmoid") {
    return dnnl::algorithm::eltwise_logistic;
  } else if (func_ == "linear") {
    return dnnl::algorithm::eltwise_linear;
  } else {
    throw std::invalid_argument("Unsupported function for oneDNN: " + func_);
  }
}

bool EWLayer_oneDNN::is_function_supported(const std::string& function) {
  return (function == "relu" || function == "tanh" || function == "sigmoid" ||
          function == "linear");
}

}  // namespace it_lab_ai