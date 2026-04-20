#include "layers/BatchNormalizationLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers_fused/ConvRelu.hpp"
#include "layers_fused/DenseNetPath.hpp"

namespace it_lab_ai {

void DenseNetPath::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  RuntimeOptions default_options;
  run(input, output, default_options);
}

void DenseNetPath::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output,
                       const RuntimeOptions& options) {

  std::vector<Tensor> temp_output(1, Tensor());
  std::vector<Tensor> temp_output1(1, Tensor());

  if (input.size() != 1) {
    throw std::runtime_error(
        "BatchNormalizationLayer: Expected 1 input tensor (X)");
  }

  const auto& x = input[0];
  const auto& input_shape = x.get_shape();

  if (input_shape.dims() < 2) {
    throw std::runtime_error(
        "BatchNormalizationLayer: Input must have at least 2 dimensions");
  }

  size_t num_channels = input_shape[1];
  validate_parameters(num_channels);

  Type expected_type = x.get_type();
  if (scale_.get_type() != expected_type || bias_.get_type() != expected_type ||
      mean_.get_type() != expected_type || var_.get_type() != expected_type) {
    throw std::runtime_error(
        "BatchNormalizationLayer: Parameter type mismatch");
  }

  switch (x.get_type()) {
    case Type::kFloat:
      batchnormrelu_impl<float>(x, temp_output[0]);
      break;
    case Type::kInt:
      batchnormrelu_impl<int>(x, temp_output[0]);
      break;
    default:
      throw std::runtime_error(
          "BatchNormalizationLayer: Unsupported input tensor type");
  }

  ConvReluLayer convrelu(stride1_, pads1_, dilations1_, kernel1_, bias1_,
                         group1_, useLegacyImpl1_);
  convrelu.run(temp_output, temp_output1, options);
  
  ConvolutionalLayer conv(stride2_, pads2_, dilations2_, kernel2_, bias2_, group2_,
                          useLegacyImpl2_);
  conv.run(temp_output1, output, options);
}

void DenseNetPath::validate_parameters(size_t num_channels) const {
  auto check_parameter = [num_channels](const Tensor& param, const char* name) {
    auto param_shape = param.get_shape();
    if (param_shape.dims() != 1 || param_shape[0] != num_channels) {
      throw std::runtime_error(
          std::string("BatchNormalizationLayer: Invalid ") + name +
          " parameter shape. Expected [" + std::to_string(num_channels) +
          "], got " + std::to_string(param_shape[0]));
    }
  };

  check_parameter(scale_, "scale");
  check_parameter(bias_, "bias");
  check_parameter(mean_, "mean");
  check_parameter(var_, "var");
}

template <typename T>
void DenseNetPath::batchnormrelu_impl(const Tensor& input,
                                      Tensor& output) const {
  const auto* scale_data = scale_.as<T>();
  const auto* bias_data = bias_.as<T>();
  const auto* mean_data = mean_.as<T>();
  const auto* var_data = var_.as<T>();
  const auto* input_data = input.as<T>();

  if (!input_data || !scale_data || !bias_data || !mean_data || !var_data) {
    throw std::runtime_error("BatchNormalizationLayer: Invalid tensor data");
  }

  const auto& shape = input.get_shape();
  size_t batch_size = shape[0];
  size_t num_channels = shape[1];
  size_t spatial_size = shape.count() / (batch_size * num_channels);

  output = Tensor(shape, input.get_type());
  auto* output_data = output.as<T>();

  if (!output_data) {
    throw std::runtime_error(
        "BatchNormalizationLayer: Failed to create output tensor");
  }

  if (!training_mode_) {
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < num_channels; ++c) {
        T scale_val = (*scale_data)[c];
        T bias1_val = (*bias_data)[c];
        T mean_val = (*mean_data)[c];
        T var_val = (*var_data)[c];

        T normalization_factor =
            static_cast<T>(1.0) /
            static_cast<T>(std::sqrt(static_cast<double>(var_val) + epsilon_));

        for (size_t i = 0; i < spatial_size; ++i) {
          size_t index = b * num_channels * spatial_size + c * spatial_size + i;
          T input_val = (*input_data)[index];
          T normalized = (input_val - mean_val) * normalization_factor;
          T temp_val = normalized * scale_val + bias1_val;
          (*output_data)[index] = temp_val > 0 ? temp_val : static_cast<T>(0);
        }
      }
    }
  } else {
    throw std::runtime_error(
        "BatchNormalizationLayer: Training mode not implemented for inference");
  }
}

template void DenseNetPath::batchnormrelu_impl<float>(const Tensor&,
                                                             Tensor&) const;

template void DenseNetPath::batchnormrelu_impl<int>(const Tensor&,
                                                           Tensor&) const;
}  // namespace it_lab_ai