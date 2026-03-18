#include "layers_oneDNN/ConvLayer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace it_lab_ai {

void ConvLayerOneDnn::run(const std::vector<Tensor>& input,
                          std::vector<Tensor>& output) {
  if (kernel_ == nullptr || bias_ == nullptr) {
    throw std::runtime_error("ConvLayerOneDnn: no kernel or bias");
  }
  if (use_legacy_) {
    run_special_conv(input, output);
    return;
  }

  validate_input(input);

  const Tensor& input_tensor = input[0];
  Type data_type = input_tensor.get_type();
  const Shape& input_shape = input_tensor.get_shape();

  bool need_reinit = !initialized_ || input_shape != last_input_shape_ ||
                     data_type != last_data_type_;
  if (need_reinit) {
    initialize_convolution(input_shape, data_type);
    last_input_shape_ = input_shape;
    last_data_type_ = data_type;
  }

  try {
    if (data_type == Type::kFloat) {
      const std::vector<float>& input_data = *input_tensor.as<float>();
      std::copy(input_data.begin(), input_data.end(),
                static_cast<float*>(src_memory_.get_data_handle()));
    } else if (data_type == Type::kInt) {
      const std::vector<int>& input_data = *input_tensor.as<int>();
      std::vector<float> float_input(input_data.size());
      std::transform(input_data.begin(), input_data.end(), float_input.begin(),
                     [](int val) { return static_cast<float>(val); });
      std::copy(float_input.begin(), float_input.end(),
                static_cast<float*>(src_memory_.get_data_handle()));
    }

    if (!bias_->empty()) {
      conv_prim_->execute(*stream_, {{DNNL_ARG_SRC, src_memory_},
                                     {DNNL_ARG_WEIGHTS, weights_memory_},
                                     {DNNL_ARG_BIAS, bias_memory_},
                                     {DNNL_ARG_DST, dst_memory_}});
    } else {
      conv_prim_->execute(*stream_, {{DNNL_ARG_SRC, src_memory_},
                                     {DNNL_ARG_WEIGHTS, weights_memory_},
                                     {DNNL_ARG_DST, dst_memory_}});
    }

    stream_->wait();

    Shape output_shape = get_output_shape(input_shape);
    create_output_tensor(output[0], output_shape, data_type, dst_memory_);

  } catch (const std::exception& e) {
    std::cerr << "oneDNN convolution execution failed: " << e.what() << '\n';
    throw;
  }
}

void ConvLayerOneDnn::create_output_tensor(Tensor& output_tensor,
                                           const Shape& output_shape,
                                           Type data_type,
                                           dnnl::memory& dst_memory) {
  size_t output_size = output_shape.count();
  auto* dst_data = static_cast<float*>(dst_memory.get_data_handle());

  if (data_type == Type::kFloat) {
    std::vector<float> output_data(output_size);
    std::copy_n(dst_data, output_size, output_data.begin());
    output_tensor = make_tensor(output_data, output_shape);
  } else if (data_type == Type::kInt) {
    std::vector<float> float_output(output_size);
    std::copy_n(dst_data, output_size, float_output.begin());

    std::vector<int> int_output(output_size);
    std::transform(float_output.begin(), float_output.end(), int_output.begin(),
                   [](float val) { return static_cast<int>(std::round(val)); });

    output_tensor = make_tensor(int_output, output_shape);
  }
}

void ConvLayerOneDnn::validate_input(const std::vector<Tensor>& input) const {
  if (input.size() != 1) {
    throw std::runtime_error(
        "ConvLayerOneDnn: Expected exactly 1 input tensor");
  }

  const Shape& input_shape = input[0].get_shape();
  const Shape& kernel_shape = kernel_->get_shape();

  if (input_shape.dims() != 4) {
    throw std::runtime_error("ConvLayerOneDnn: Input must be 4D (NCHW format)");
  }

  if (kernel_shape.dims() != 4) {
    throw std::runtime_error("ConvLayerOneDnn: Kernel must be 4D");
  }

  if (is_depthwise_convolution()) {
    validate_depthwise_input(input);
    return;
  }

  size_t in_channels = input_shape[1];
  size_t kernel_in_channels = kernel_shape[1];

  if (group_ > 1) {
    if (in_channels % group_ != 0) {
      throw std::runtime_error(
          "ConvLayerOneDnn: Input channels must be divisible by group");
    }
    if (kernel_in_channels != in_channels / group_) {
      throw std::runtime_error(
          "ConvLayerOneDnn: Kernel input channels don't match group "
          "configuration");
    }
  } else {
    if (in_channels != kernel_in_channels) {
      throw std::runtime_error(
          "ConvLayerOneDnn: Input and kernel channels don't match");
    }
  }

  Type data_type = input[0].get_type();
  if (data_type != Type::kFloat && data_type != Type::kInt) {
    throw std::runtime_error(
        "ConvLayerOneDnn supports only float and int tensors");
  }
}

void ConvLayerOneDnn::validate_depthwise_input(
    const std::vector<Tensor>& input) const {
  const Shape& input_shape = input[0].get_shape();
  const Shape& kernel_shape = kernel_->get_shape();

  size_t in_channels = input_shape[1];
  size_t kernel_out_channels = kernel_shape[0];
  size_t kernel_in_channels = kernel_shape[1];

  if (kernel_out_channels != in_channels || kernel_in_channels != 1) {
    throw std::runtime_error("Invalid kernel shape for depthwise convolution");
  }

  Type data_type = input[0].get_type();
  if (data_type != Type::kFloat && data_type != Type::kInt) {
    throw std::runtime_error(
        "ConvLayerOneDnn supports only float and int tensors");
  }
}

bool ConvLayerOneDnn::is_depthwise_convolution() const {
  const Shape& kernel_shape = kernel_->get_shape();
  return (group_ == kernel_shape[0]);
}

void ConvLayerOneDnn::initialize_convolution(const Shape& input_shape,
                                             Type data_type) {
  try {
    engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    stream_ = std::make_unique<dnnl::stream>(*engine_);

    const size_t in_channels = input_shape[1];
    bool is_depthwise = (group_ > 1 && group_ == in_channels);

    dnnl::memory::dims src_dims = shape_to_dims(input_shape);
    dnnl::memory::dims dst_dims = get_output_dims(input_shape);
    dnnl::memory::dims strides = {static_cast<dnnl::memory::dim>(stride_),
                                  static_cast<dnnl::memory::dim>(stride_)};
    dnnl::memory::dims padding = {static_cast<dnnl::memory::dim>(pads_),
                                  static_cast<dnnl::memory::dim>(pads_)};
    dnnl::memory::dims dilation = {
        static_cast<dnnl::memory::dim>(dilations_ - 1),
        static_cast<dnnl::memory::dim>(dilations_ - 1)};

    dnnl::memory::data_type dnnl_data_type = dnnl::memory::data_type::f32;

    auto src_md = dnnl::memory::desc(src_dims, dnnl_data_type,
                                     dnnl::memory::format_tag::any);
    auto dst_md = dnnl::memory::desc(dst_dims, dnnl_data_type,
                                     dnnl::memory::format_tag::any);

    const auto [kernel_dims, weights_format] =
        [&]() -> std::pair<dnnl::memory::dims, dnnl::memory::format_tag> {
      if (is_depthwise) {
        return {{static_cast<dnnl::memory::dim>(group_), 1, 1,
                 static_cast<dnnl::memory::dim>(kernel_->get_shape()[2]),
                 static_cast<dnnl::memory::dim>(kernel_->get_shape()[3])},
                dnnl::memory::format_tag::goihw};
      }
      if (group_ > 1) {
        return {
            {static_cast<dnnl::memory::dim>(group_),
             static_cast<dnnl::memory::dim>(kernel_->get_shape()[0] / group_),
             static_cast<dnnl::memory::dim>(kernel_->get_shape()[1]),
             static_cast<dnnl::memory::dim>(kernel_->get_shape()[2]),
             static_cast<dnnl::memory::dim>(kernel_->get_shape()[3])},
            dnnl::memory::format_tag::goihw};
      }
      const auto& k_shape = kernel_->get_shape();
      return {{static_cast<dnnl::memory::dim>(k_shape[0]),
               static_cast<dnnl::memory::dim>(k_shape[1]),
               static_cast<dnnl::memory::dim>(k_shape[2]),
               static_cast<dnnl::memory::dim>(k_shape[3])},
              dnnl::memory::format_tag::oihw};
    }();

    auto weights_md =
        dnnl::memory::desc(kernel_dims, dnnl_data_type, weights_format);

    dnnl::memory::desc bias_md;
    bool has_bias = !bias_->empty();
    if (!bias_->empty()) {
      const size_t bias_size = (is_depthwise || group_ == 1)
                                   ? static_cast<size_t>(kernel_dims[0])
                                   : kernel_->get_shape()[0];

      bias_md =
          dnnl::memory::desc({static_cast<dnnl::memory::dim>(bias_size)},
                             dnnl_data_type, dnnl::memory::format_tag::any);
    }

    dnnl::convolution_forward::primitive_desc conv_pd =
        has_bias ? dnnl::convolution_forward::primitive_desc(
                       *engine_, dnnl::prop_kind::forward_inference,
                       dnnl::algorithm::convolution_direct, src_md, weights_md,
                       bias_md, dst_md, strides, dilation, padding, padding)
                 : dnnl::convolution_forward::primitive_desc(
                       *engine_, dnnl::prop_kind::forward_inference,
                       dnnl::algorithm::convolution_direct, src_md, weights_md,
                       dst_md, strides, dilation, padding, padding);

    src_memory_ = dnnl::memory(conv_pd.src_desc(), *engine_);
    weights_memory_ = dnnl::memory(conv_pd.weights_desc(), *engine_);
    dst_memory_ = dnnl::memory(conv_pd.dst_desc(), *engine_);
    if (!bias_->empty()) {
      bias_memory_ = dnnl::memory(conv_pd.bias_desc(), *engine_);
    }

    fill_memory_with_tensor(weights_memory_, *kernel_, data_type);
    if (!bias_->empty()) {
      fill_memory_with_tensor(bias_memory_, *bias_, data_type);
    }

    conv_prim_ = std::make_unique<dnnl::convolution_forward>(conv_pd);
    initialized_ = true;

  } catch (const dnnl::error& e) {
    std::cerr << "oneDNN specific error: " << e.what()
              << ", status: " << e.status << '\n';
    throw;
  } catch (const std::exception& e) {
    std::cerr << "oneDNN convolution initialization failed: " << e.what()
              << '\n';
    throw;
  }
}

void ConvLayerOneDnn::fill_memory_with_tensor(dnnl::memory& memory,
                                              const Tensor& tensor,
                                              Type data_type) {
  if (data_type == Type::kFloat) {
    const std::vector<float>& data = *tensor.as<float>();
    std::copy(data.begin(), data.end(),
              static_cast<float*>(memory.get_data_handle()));
  } else if (data_type == Type::kInt) {
    const std::vector<int>& data = *tensor.as<int>();
    std::vector<float> float_data(data.size());
    std::transform(data.begin(), data.end(), float_data.begin(),
                   [](int val) { return static_cast<float>(val); });
    std::copy(float_data.begin(), float_data.end(),
              static_cast<float*>(memory.get_data_handle()));
  }
}

dnnl::memory::dims ConvLayerOneDnn::get_kernel_dims() const {
  const Shape& kernel_shape = kernel_->get_shape();

  dnnl::memory::dims dims;
  for (size_t i = 0; i < kernel_shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(kernel_shape[i]));
  }

  return dims;
}

Shape ConvLayerOneDnn::get_output_shape(const Shape& input_shape) const {
  const Shape& kernel_shape = kernel_->get_shape();

  const auto [kernel_out_channels, kernel_height,
              kernel_width] = [&]() -> std::tuple<size_t, size_t, size_t> {
    if (use_legacy_ ||
        (kernel_shape.dims() == 4 && kernel_shape[3] > kernel_shape[2])) {
      return {kernel_shape[3], kernel_shape[0], kernel_shape[1]};
    }
    return {kernel_shape[0], kernel_shape[2], kernel_shape[3]};
  }();

  size_t batch_size = input_shape[0];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];

  size_t effective_kernel_height = (kernel_height - 1) * dilations_ + 1;
  size_t effective_kernel_width = (kernel_width - 1) * dilations_ + 1;

  size_t output_height =
      (input_height + 2 * pads_ - effective_kernel_height) / stride_ + 1;
  size_t output_width =
      (input_width + 2 * pads_ - effective_kernel_width) / stride_ + 1;

  return Shape({batch_size, kernel_out_channels, output_height, output_width});
}

void ConvLayerOneDnn::initialize_special_conv(const Shape& input_shape,
                                              Type data_type) {
  try {
    engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    stream_ = std::make_unique<dnnl::stream>(*engine_);

    dnnl::memory::dims src_dims = shape_to_dims(input_shape);

    Shape output_shape = get_output_shape(input_shape);
    dnnl::memory::dims dst_dims = shape_to_dims(output_shape);

    dnnl::memory::dims strides = {static_cast<dnnl::memory::dim>(stride_),
                                  static_cast<dnnl::memory::dim>(stride_)};
    dnnl::memory::dims padding = {static_cast<dnnl::memory::dim>(pads_),
                                  static_cast<dnnl::memory::dim>(pads_)};
    dnnl::memory::dims dilation = {
        static_cast<dnnl::memory::dim>(dilations_ - 1),
        static_cast<dnnl::memory::dim>(dilations_ - 1)};

    dnnl::memory::data_type dt = dnnl::memory::data_type::f32;

    auto src_md =
        dnnl::memory::desc(src_dims, dt, dnnl::memory::format_tag::nchw);
    auto dst_md =
        dnnl::memory::desc(dst_dims, dt, dnnl::memory::format_tag::nchw);

    const auto& k_shape = kernel_->get_shape();

    dnnl::memory::dims weights_dims = {
        static_cast<dnnl::memory::dim>(k_shape[3]),
        static_cast<dnnl::memory::dim>(k_shape[2]),
        static_cast<dnnl::memory::dim>(k_shape[0]),
        static_cast<dnnl::memory::dim>(k_shape[1])};

    auto weights_md =
        dnnl::memory::desc(weights_dims, dt, dnnl::memory::format_tag::oihw);

    dnnl::memory::desc bias_md;
    bool has_bias = !bias_->empty();
    if (has_bias) {
      bias_md = dnnl::memory::desc(
          {static_cast<dnnl::memory::dim>(bias_->get_shape()[0])}, dt,
          dnnl::memory::format_tag::a);
    }

    dnnl::convolution_forward::primitive_desc conv_pd =
        has_bias ? dnnl::convolution_forward::primitive_desc(
                       *engine_, dnnl::prop_kind::forward_inference,
                       dnnl::algorithm::convolution_direct, src_md, weights_md,
                       bias_md, dst_md, strides, dilation, padding, padding)
                 : dnnl::convolution_forward::primitive_desc(
                       *engine_, dnnl::prop_kind::forward_inference,
                       dnnl::algorithm::convolution_direct, src_md, weights_md,
                       dst_md, strides, dilation, padding, padding);

    src_memory_ = dnnl::memory(conv_pd.src_desc(), *engine_);
    weights_memory_ = dnnl::memory(conv_pd.weights_desc(), *engine_);
    dst_memory_ = dnnl::memory(conv_pd.dst_desc(), *engine_);

    if (has_bias) {
      bias_memory_ = dnnl::memory(conv_pd.bias_desc(), *engine_);
    }

    if (data_type == Type::kFloat) {
      const std::vector<float>& kernel_data = *kernel_->as<float>();

      size_t kh = k_shape[0];
      size_t kw = k_shape[1];
      size_t kic = k_shape[2];
      size_t koc = k_shape[3];

      std::vector<float> reordered(koc * kic * kh * kw);

      for (size_t oc = 0; oc < koc; ++oc) {
        for (size_t ic = 0; ic < kic; ++ic) {
          for (size_t h = 0; h < kh; ++h) {
            for (size_t w = 0; w < kw; ++w) {
              size_t src_idx = ((h * kw + w) * kic + ic) * koc + oc;
              size_t dst_idx = ((oc * kic + ic) * kh + h) * kw + w;
              reordered[dst_idx] = kernel_data[src_idx];
            }
          }
        }
      }

      std::memcpy(weights_memory_.get_data_handle(), reordered.data(),
                  reordered.size() * sizeof(float));

    } else if (data_type == Type::kInt) {
      const std::vector<int>& kernel_data_int = *kernel_->as<int>();
      size_t kh = k_shape[0];
      size_t kw = k_shape[1];
      size_t kic = k_shape[2];
      size_t koc = k_shape[3];

      std::vector<float> reordered(koc * kic * kh * kw);
      for (size_t oc = 0; oc < koc; ++oc) {
        for (size_t ic = 0; ic < kic; ++ic) {
          for (size_t h = 0; h < kh; ++h) {
            for (size_t w = 0; w < kw; ++w) {
              size_t src_idx = ((h * kw + w) * kic + ic) * koc + oc;
              size_t dst_idx = ((oc * kic + ic) * kh + h) * kw + w;
              reordered[dst_idx] = static_cast<float>(kernel_data_int[src_idx]);
            }
          }
        }
      }

      std::memcpy(weights_memory_.get_data_handle(), reordered.data(),
                  reordered.size() * sizeof(float));
    }

    if (has_bias) {
      if (data_type == Type::kFloat) {
        const std::vector<float>& bias_data = *bias_->as<float>();
        std::memcpy(bias_memory_.get_data_handle(), bias_data.data(),
                    bias_data.size() * sizeof(float));
      } else if (data_type == Type::kInt) {
        const std::vector<int>& bias_data_int = *bias_->as<int>();
        std::vector<float> float_bias(bias_data_int.size());
        std::transform(bias_data_int.begin(), bias_data_int.end(),
                       float_bias.begin(),
                       [](int val) { return static_cast<float>(val); });
        std::memcpy(bias_memory_.get_data_handle(), float_bias.data(),
                    float_bias.size() * sizeof(float));
      }
    }

    conv_prim_ = std::make_unique<dnnl::convolution_forward>(conv_pd);
    initialized_ = true;

  } catch (const dnnl::error& e) {
    std::cerr << "oneDNN error: " << e.what() << ", status: " << e.status
              << '\n';
    throw;
  } catch (const std::exception& e) {
    std::cerr << "Special conv initialization failed: " << e.what() << '\n';
    throw;
  }
}

void ConvLayerOneDnn::run_special_conv(const std::vector<Tensor>& input,
                                       std::vector<Tensor>& output) {
  const Tensor& input_tensor = input[0];
  Type data_type = input_tensor.get_type();
  const Shape& input_shape = input_tensor.get_shape();

  if (!initialized_ || input_shape != last_input_shape_ ||
      data_type != last_data_type_) {
    initialize_special_conv(input_shape, data_type);
    last_input_shape_ = input_shape;
    last_data_type_ = data_type;
  }

  if (data_type == Type::kFloat) {
    const std::vector<float>& input_data = *input_tensor.as<float>();
    std::memcpy(src_memory_.get_data_handle(), input_data.data(),
                input_data.size() * sizeof(float));
  } else if (data_type == Type::kInt) {
    const std::vector<int>& input_data = *input_tensor.as<int>();
    std::vector<float> float_input(input_data.size());
    std::transform(input_data.begin(), input_data.end(), float_input.begin(),
                   [](int val) { return static_cast<float>(val); });
    std::memcpy(src_memory_.get_data_handle(), float_input.data(),
                float_input.size() * sizeof(float));
  } else {
    throw std::runtime_error("Unsupported input type in run_special_conv");
  }

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, src_memory_},
      {DNNL_ARG_WEIGHTS, weights_memory_},
      {DNNL_ARG_DST, dst_memory_}};

  if (!bias_->empty()) {
    args[DNNL_ARG_BIAS] = bias_memory_;
  }

  conv_prim_->execute(*stream_, args);
  stream_->wait();

  Shape output_shape = get_output_shape(input_shape);

  if (data_type == Type::kFloat) {
    size_t output_size = dst_memory_.get_desc().get_size() / sizeof(float);
    std::vector<float> output_data(output_size);
    std::memcpy(output_data.data(), dst_memory_.get_data_handle(),
                output_data.size() * sizeof(float));
    output[0] = make_tensor(output_data, output_shape);
  } else if (data_type == Type::kInt) {
    std::vector<float> tmp(dst_memory_.get_desc().get_size() / sizeof(float));
    std::memcpy(tmp.data(), dst_memory_.get_data_handle(),
                tmp.size() * sizeof(float));
    std::vector<int> output_data(tmp.size());
    std::transform(tmp.begin(), tmp.end(), output_data.begin(),
                   [](float val) { return static_cast<int>(val); });
    output[0] = make_tensor(output_data, output_shape);
  }
}

template <typename T>
std::vector<T> ConvLayerOneDnn::reorder_hwio_to_oihw(const Tensor& kernel) {
  size_t kh = kernel.get_shape()[0];
  size_t kw = kernel.get_shape()[1];
  size_t kic = kernel.get_shape()[2];
  size_t koc = kernel.get_shape()[3];

  std::vector<T> result(koc * kic * kh * kw);

  size_t idx = 0;
  for (size_t oc = 0; oc < koc; oc++) {
    for (size_t ic = 0; ic < kic; ic++) {
      for (size_t h = 0; h < kh; h++) {
        for (size_t w = 0; w < kw; w++) {
          result[idx++] = kernel.get<T>({h, w, ic, oc});
        }
      }
    }
  }
  return result;
}

}  // namespace it_lab_ai
