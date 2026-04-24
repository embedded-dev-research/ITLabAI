#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layers/ConvLayer.hpp"
#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

template <typename T>
void relu(Tensor& t) {
  Shape sh = t.get_shape();
  for (size_t i = 0; i < sh.count(); i++) {
    if ((*t.as<T>())[i] < 0) {
      (*t.as<T>())[i] = 0;
    }
  }
}

class ConvReluLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  std::shared_ptr<Tensor> kernel_;
  std::shared_ptr<Tensor> bias_;
  size_t group_;
  bool useLegacyImpl_;

 public:
  ConvReluLayer() : Layer(kConvRelu), kernel_(nullptr), bias_(nullptr) {
    stride_ = 0;
    pads_ = 0;
    dilations_ = 0;
  }
  ConvReluLayer(size_t step, size_t pads, size_t dilations,
                const Tensor& kernel, const Tensor& bias = Tensor(),
                size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvRelu),
        kernel_(std::make_shared<Tensor>(kernel)),
        bias_(std::make_shared<Tensor>(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }
  ConvReluLayer(size_t step, size_t pads, size_t dilations,
                std::shared_ptr<Tensor> kernel,
                std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(),
                size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvRelu), kernel_(std::move(kernel)), bias_(std::move(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }
  explicit ConvReluLayer(const std::shared_ptr<ConvolutionalLayer>& conv)
      : Layer(kConvRelu) {
    auto numerics = conv->getNumericParams();
    auto tensors = conv->getTensorParams();
    stride_ = numerics[0];
    pads_ = numerics[1];
    dilations_ = numerics[2];
    group_ = numerics[3];
    kernel_ = tensors[0];
    bias_ = tensors[1];
    useLegacyImpl_ = conv->getLegacyImplBool();
  }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
           const RuntimeOptions& options) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return *kernel_; }
#endif
};

// NCHW -> NCHW only
template <typename ValueType>
void Conv4DRelu(const Tensor& input, const Tensor& kernel_, const Tensor& bias_,
                Tensor& output, size_t stride_, size_t pads_, size_t group_,
                size_t dilations_, ParBackend backend = ParBackend::kSeq) {
  size_t batch_size = input.get_shape()[0];
  size_t in_channels = input.get_shape()[1];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];

  size_t out_channels = kernel_.get_shape()[0];
  size_t kernel_in_channels = kernel_.get_shape()[1];
  size_t kernel_height = kernel_.get_shape()[2];
  size_t kernel_width = kernel_.get_shape()[3];

  if (group_ > 1) {
    if (in_channels % group_ != 0 || out_channels % group_ != 0) {
      throw std::runtime_error("Channels must be divisible by group");
    }
    if (kernel_in_channels != in_channels / group_) {
      throw std::runtime_error(
          "Kernel input channels don't match group configuration");
    }
  }

  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
                                           pads_, dilations_);
  size_t out_width =
      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          in_height + 2 * pads_,
          std::vector<std::vector<ValueType>>(
              in_width + 2 * pads_, std::vector<ValueType>(in_channels, 0))));

  parallel::Options options;
  options.backend = backend;

  parallel::parallel_for(
      batch_size,
      [&](size_t b) {
        for (size_t h = 0; h < in_height; ++h) {
          for (size_t w = 0; w < in_width; ++w) {
            for (size_t c = 0; c < in_channels; ++c) {
              padded_input[b][h + pads_][w + pads_][c] =
                  input.get<ValueType>({b, c, h, w});
            }
          }
        }
      },
      options);

  size_t dilated_kernel_height = (kernel_height - 1) * dilations_ + 1;
  size_t dilated_kernel_width = (kernel_width - 1) * dilations_ + 1;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel(
      out_channels, std::vector<std::vector<std::vector<ValueType>>>(
                        kernel_in_channels,
                        std::vector<std::vector<ValueType>>(
                            dilated_kernel_height,
                            std::vector<ValueType>(dilated_kernel_width, 0))));

  parallel::parallel_for(
      out_channels,
      [&](size_t oc) {
        for (size_t ic = 0; ic < kernel_in_channels; ++ic) {
          for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
              dil_kernel[oc][ic][kh * dilations_][kw * dilations_] =
                  kernel_.get<ValueType>({oc, ic, kh, kw});
            }
          }
        }
      },
      options);

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          out_channels, std::vector<std::vector<ValueType>>(
                            out_height, std::vector<ValueType>(out_width, 0))));

  size_t total_work = batch_size * out_channels;
  parallel::parallel_for(
      total_work,
      [&](size_t idx) {
        size_t b = idx / out_channels;
        size_t oc = idx % out_channels;

        for (size_t oh = 0; oh < out_height; ++oh) {
          for (size_t ow = 0; ow < out_width; ++ow) {
            ValueType value = 0;
            size_t h_start = oh * stride_;
            size_t w_start = ow * stride_;

            size_t group = (group_ > 1) ? oc / (out_channels / group_) : 0;
            size_t group_start_channel = group * (in_channels / group_);
            size_t group_end_channel = (group + 1) * (in_channels / group_);

            for (size_t ic = group_start_channel; ic < group_end_channel;
                 ++ic) {
              size_t kernel_ic = ic - group_start_channel;

              for (size_t kh = 0; kh < dilated_kernel_height; ++kh) {
                for (size_t kw = 0; kw < dilated_kernel_width; ++kw) {
                  size_t h_index = h_start + kh;
                  size_t w_index = w_start + kw;

                  if (h_index < padded_input[b].size() &&
                      w_index < padded_input[b][h_index].size()) {
                    value += padded_input[b][h_index][w_index][ic] *
                             dil_kernel[oc][kernel_ic][kh][kw];
                  }
                }
              }
            }

            if (!bias_.empty() && oc < bias_.get_shape()[0]) {
              value += bias_.get<ValueType>({oc});
            }

            output_tensor[b][oc][oh][ow] = value > 0 ? value : 0;
          }
        }
      },
      options);

  Shape output_shape({batch_size, out_channels, out_height, out_width});
  std::vector<ValueType> flat_output(batch_size * out_channels * out_height *
                                     out_width);

  parallel::parallel_for(
      batch_size,
      [&](size_t b) {
        size_t base_idx = b * out_channels * out_height * out_width;
        for (size_t oc = 0; oc < out_channels; ++oc) {
          for (size_t h = 0; h < out_height; ++h) {
            for (size_t w = 0; w < out_width; ++w) {
              flat_output[base_idx++] = output_tensor[b][oc][h][w];
            }
          }
        }
      },
      options);

  output = make_tensor<ValueType>(flat_output, output_shape);
}

template <typename ValueType>
void DepthwiseConv4DRelu(const Tensor& input, const Tensor& kernel_,
                         const Tensor& bias_, Tensor& output, size_t stride_,
                         size_t pads_, size_t dilations_,
                         ParBackend backend = ParBackend::kSeq) {
  size_t batch_size = input.get_shape()[0];
  size_t channels = input.get_shape()[1];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];

  size_t kernel_out_channels = kernel_.get_shape()[0];
  size_t kernel_in_channels = kernel_.get_shape()[1];
  size_t kernel_height = kernel_.get_shape()[2];
  size_t kernel_width = kernel_.get_shape()[3];

  if (kernel_out_channels != channels || kernel_in_channels != 1) {
    throw std::runtime_error("Invalid kernel shape for depthwise convolution");
  }

  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
                                           pads_, dilations_);
  size_t out_width =
      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);

  Tensor output_tensor(Shape({batch_size, channels, out_height, out_width}),
                       input.get_type());

  parallel::Options options;
  options.backend = backend;

  size_t total_work = batch_size * channels;

  parallel::parallel_for(
      total_work,
      [&](size_t idx) {
        size_t b = idx / channels;
        size_t c = idx % channels;

        for (size_t oh = 0; oh < out_height; ++oh) {
          for (size_t ow = 0; ow < out_width; ++ow) {
            ValueType sum = 0;

            for (size_t kh = 0; kh < kernel_height; ++kh) {
              for (size_t kw = 0; kw < kernel_width; ++kw) {
                size_t ih = oh * stride_ + kh * dilations_;
                size_t iw = ow * stride_ + kw * dilations_;

                if (ih >= pads_ && iw >= pads_ && (ih - pads_) < in_height &&
                    (iw - pads_) < in_width) {
                  auto input_val =
                      input.get<ValueType>({b, c, ih - pads_, iw - pads_});
                  auto kernel_val = kernel_.get<ValueType>({c, 0, kh, kw});
                  sum += input_val * kernel_val;
                }
              }
            }

            if (!bias_.empty() && c < bias_.get_shape()[0]) {
              sum += bias_.get<ValueType>({c});
            }

            output_tensor.set<ValueType>({b, c, oh, ow}, sum > 0 ? sum : 0);
          }
        }
      },
      options);

  output = output_tensor;
}

// NCHW -> NCHW only (Legacy version)
template <typename ValueType>
void Conv4D_LegacyRelu(const Tensor& input, const Tensor& kernel_,
                       const Tensor& bias_, Tensor& output, size_t stride_,
                       size_t pads_, size_t dilations_,
                       ParBackend backend = ParBackend::kSeq) {
  size_t batch_size = input.get_shape()[0];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];
  size_t in_channels = input.get_shape()[1];

  size_t kernel_height = kernel_.get_shape()[0];
  size_t kernel_width = kernel_.get_shape()[1];
  size_t kernel_in_channels = kernel_.get_shape()[2];
  size_t kernel_out_channels = kernel_.get_shape()[3];

  parallel::Options options;
  options.backend = backend;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          in_height + 2 * pads_,
          std::vector<std::vector<ValueType>>(
              in_width + 2 * pads_, std::vector<ValueType>(in_channels, 0))));

  parallel::parallel_for(
      batch_size,
      [&](size_t b) {
        for (size_t h = 0; h < in_height; ++h) {
          for (size_t w = 0; w < in_width; ++w) {
            for (size_t c = 0; c < in_channels; ++c) {
              padded_input[b][h + pads_][w + pads_][c] =
                  input.get<ValueType>({b, c, h, w});
            }
          }
        }
      },
      options);

  size_t dilated_kernel_height = kernel_height * dilations_ + 1 - dilations_;
  size_t dilated_kernel_width = kernel_width * dilations_ + 1 - dilations_;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel(
      dilated_kernel_height,
      std::vector<std::vector<std::vector<ValueType>>>(
          dilated_kernel_width,
          std::vector<std::vector<ValueType>>(
              kernel_in_channels,
              std::vector<ValueType>(kernel_out_channels, 0))));

  parallel::parallel_for(
      kernel_out_channels,
      [&](size_t b) {
        for (size_t h = 0; h < kernel_height; ++h) {
          for (size_t w = 0; w < kernel_width; ++w) {
            for (size_t c = 0; c < kernel_in_channels; ++c) {
              dil_kernel[h * dilations_][w * dilations_][c][b] =
                  kernel_.get<ValueType>({h, w, c, b});
            }
          }
        }
      },
      options);

  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
                                           pads_, dilations_);
  size_t out_width =
      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                      kernel_out_channels,
                      std::vector<std::vector<ValueType>>(
                          out_height, std::vector<ValueType>(out_width, 0))));

  size_t total_work = batch_size * kernel_out_channels;

  parallel::parallel_for(
      total_work,
      [&](size_t idx) {
        size_t b = idx / kernel_out_channels;
        size_t c = idx % kernel_out_channels;

        for (size_t i = 0; i < out_height; i += stride_) {
          for (size_t j = 0; j < out_width; j += stride_) {
            ValueType value = 0;
            for (size_t ic = 0; ic < in_channels; ++ic) {
              for (size_t h = 0; h < dilated_kernel_height; ++h) {
                for (size_t w = 0; w < dilated_kernel_width; ++w) {
                  value += padded_input[b][i + h][j + w][ic] *
                           dil_kernel[h][w][ic][c];
                }
              }
            }
            if (!bias_.empty()) {
              output_tensor[b][c][i][j] =
                  value + (*bias_.as<ValueType>())[c] > 0
                      ? (value + (*bias_.as<ValueType>())[c])
                      : 0;
            } else {
              output_tensor[b][c][i][j] = value > 0 ? value : 0;
            }
          }
        }
      },
      options);

  Shape sh({batch_size, kernel_out_channels, out_height, out_width});
  std::vector<ValueType> one_d_vector(batch_size * out_height * out_width *
                                      kernel_out_channels);

  parallel::parallel_for(
      batch_size,
      [&](size_t i) {
        size_t base_idx = i * kernel_out_channels * out_height * out_width;
        for (size_t l = 0; l < kernel_out_channels; ++l) {
          for (size_t j = 0; j < out_height; ++j) {
            for (size_t k = 0; k < out_width; ++k) {
              one_d_vector[base_idx++] = output_tensor[i][l][j][k];
            }
          }
        }
      },
      options);

  output = make_tensor<ValueType>(one_d_vector, sh);
}

//// NCHW -> NCHW only
//template <typename ValueType>
//void Conv4DRelu(const Tensor& input, const Tensor& kernel_, const Tensor& bias_,
//                Tensor& output, size_t stride_, size_t pads_, size_t group_,
//                size_t dilations_, ParBackend backend = ParBackend::kSeq) {
//  size_t batch_size = input.get_shape()[0];
//  size_t in_channels = input.get_shape()[1];
//  size_t in_height = input.get_shape()[2];
//  size_t in_width = input.get_shape()[3];
//
//  size_t out_channels = kernel_.get_shape()[0];
//  size_t kernel_in_channels = kernel_.get_shape()[1];
//  size_t kernel_height = kernel_.get_shape()[2];
//  size_t kernel_width = kernel_.get_shape()[3];
//
//  if (group_ > 1) {
//    if (in_channels % group_ != 0 || out_channels % group_ != 0) {
//      throw std::runtime_error("Channels must be divisible by group");
//    }
//    if (kernel_in_channels != in_channels / group_) {
//      throw std::runtime_error(
//          "Kernel input channels don't match group configuration");
//    }
//  }
//
//  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
//                                           pads_, dilations_);
//  size_t out_width =
//      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);
//
//  parallel::Options options;
//  options.backend = backend;
//
//  const auto& input_data = *input.as<ValueType>();
//  const auto& kernel_data = *kernel_.as<ValueType>();
//  const std::vector<ValueType>* bias_data = nullptr;
//  if (!bias_.empty()) {
//    bias_data = bias_.as<ValueType>();
//  }
//
//  const size_t input_channel_stride = in_height * in_width;
//  const size_t input_batch_stride = in_channels * input_channel_stride;
//  const size_t kernel_channel_stride = kernel_height * kernel_width;
//  const size_t kernel_output_stride =
//      kernel_in_channels * kernel_channel_stride;
//  const size_t output_channel_stride = out_height * out_width;
//  const size_t output_batch_stride = out_channels * output_channel_stride;
//  const size_t in_channels_per_group = in_channels / group_;
//  const size_t out_channels_per_group = out_channels / group_;
//  const bool collapsed_kernel = dilations_ == 0;
//
//  Shape output_shape({batch_size, out_channels, out_height, out_width});
//  std::vector<ValueType> flat_output(output_shape.count(), 0);
//  size_t total_work = batch_size * out_channels;
//  parallel::parallel_for(
//      total_work,
//      [&](size_t idx) {
//        size_t b = idx / out_channels;
//        size_t oc = idx % out_channels;
//        size_t input_batch_base = b * input_batch_stride;
//        size_t output_base =
//            b * output_batch_stride + oc * output_channel_stride;
//        size_t group = (group_ > 1) ? oc / out_channels_per_group : 0;
//        size_t group_start_channel = group * in_channels_per_group;
//        size_t group_end_channel = group_start_channel + in_channels_per_group;
//        size_t kernel_oc_base = oc * kernel_output_stride;
//        auto bias_value = ValueType{};
//        if (bias_data != nullptr && oc < bias_data->size()) {
//          bias_value = (*bias_data)[oc];
//        }
//
//        for (size_t oh = 0; oh < out_height; ++oh) {
//          std::ptrdiff_t input_h_base =
//              static_cast<std::ptrdiff_t>(oh * stride_) -
//              static_cast<std::ptrdiff_t>(pads_);
//          for (size_t ow = 0; ow < out_width; ++ow) {
//            ValueType value = bias_value;
//            std::ptrdiff_t input_w_base =
//                static_cast<std::ptrdiff_t>(ow * stride_) -
//                static_cast<std::ptrdiff_t>(pads_);
//            size_t output_idx = output_base + oh * out_width + ow;
//
//            for (size_t ic = group_start_channel; ic < group_end_channel;
//                 ++ic) {
//              size_t kernel_ic = ic - group_start_channel;
//              size_t input_channel_base =
//                  input_batch_base + ic * input_channel_stride;
//              size_t kernel_ic_base =
//                  kernel_oc_base + kernel_ic * kernel_channel_stride;
//
//              if (collapsed_kernel) {
//                if (input_h_base >= 0 &&
//                    input_h_base < static_cast<std::ptrdiff_t>(in_height) &&
//                    input_w_base >= 0 &&
//                    input_w_base < static_cast<std::ptrdiff_t>(in_width)) {
//                  size_t input_idx =
//                      input_channel_base +
//                      static_cast<size_t>(input_h_base) * in_width +
//                      static_cast<size_t>(input_w_base);
//                  size_t kernel_idx =
//                      kernel_ic_base + kernel_channel_stride - 1;
//                  value += input_data[input_idx] * kernel_data[kernel_idx];
//                }
//                continue;
//              }
//
//              for (size_t kh = 0; kh < kernel_height; ++kh) {
//                std::ptrdiff_t input_h =
//                    input_h_base + static_cast<std::ptrdiff_t>(kh * dilations_);
//                if (input_h < 0 ||
//                    input_h >= static_cast<std::ptrdiff_t>(in_height)) {
//                  continue;
//                }
//
//                size_t input_row_base = input_channel_base +
//                                        static_cast<size_t>(input_h) * in_width;
//                size_t kernel_row_base = kernel_ic_base + kh * kernel_width;
//
//                for (size_t kw = 0; kw < kernel_width; ++kw) {
//                  std::ptrdiff_t input_w =
//                      input_w_base +
//                      static_cast<std::ptrdiff_t>(kw * dilations_);
//                  if (input_w < 0 ||
//                      input_w >= static_cast<std::ptrdiff_t>(in_width)) {
//                    continue;
//                  }
//
//                  value += input_data[input_row_base +
//                                      static_cast<size_t>(input_w)] *
//                           kernel_data[kernel_row_base + kw];
//                }
//              }
//            }
//
//            flat_output[output_idx] = value > 0 ? value : 0;
//          }
//        }
//      },
//      options);
//
//  output = make_tensor<ValueType>(flat_output, output_shape);
//}
//
//template <typename ValueType>
//void DepthwiseConv4DRelu(const Tensor& input, const Tensor& kernel_,
//                         const Tensor& bias_, Tensor& output, size_t stride_,
//                         size_t pads_, size_t dilations_,
//                         ParBackend backend = ParBackend::kSeq) {
//  size_t batch_size = input.get_shape()[0];
//  size_t channels = input.get_shape()[1];
//  size_t in_height = input.get_shape()[2];
//  size_t in_width = input.get_shape()[3];
//
//  size_t kernel_out_channels = kernel_.get_shape()[0];
//  size_t kernel_in_channels = kernel_.get_shape()[1];
//  size_t kernel_height = kernel_.get_shape()[2];
//  size_t kernel_width = kernel_.get_shape()[3];
//
//  if (kernel_out_channels != channels || kernel_in_channels != 1) {
//    throw std::runtime_error("Invalid kernel shape for depthwise convolution");
//  }
//
//  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
//                                           pads_, dilations_);
//  size_t out_width =
//      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);
//
//  Tensor output_tensor(Shape({batch_size, channels, out_height, out_width}),
//                       input.get_type());
//
//  parallel::Options options;
//  options.backend = backend;
//
//  size_t total_work = batch_size * channels;
//
//  parallel::parallel_for(
//      total_work,
//      [&](size_t idx) {
//        size_t b = idx / channels;
//        size_t c = idx % channels;
//
//        for (size_t oh = 0; oh < out_height; ++oh) {
//          for (size_t ow = 0; ow < out_width; ++ow) {
//            ValueType sum = 0;
//
//            for (size_t kh = 0; kh < kernel_height; ++kh) {
//              for (size_t kw = 0; kw < kernel_width; ++kw) {
//                size_t ih = oh * stride_ + kh * dilations_;
//                size_t iw = ow * stride_ + kw * dilations_;
//
//                if (ih >= pads_ && iw >= pads_ && (ih - pads_) < in_height &&
//                    (iw - pads_) < in_width) {
//                  auto input_val =
//                      input.get<ValueType>({b, c, ih - pads_, iw - pads_});
//                  auto kernel_val = kernel_.get<ValueType>({c, 0, kh, kw});
//                  sum += input_val * kernel_val;
//                }
//              }
//            }
//
//            if (!bias_.empty() && c < bias_.get_shape()[0]) {
//              sum += bias_.get<ValueType>({c});
//            }
//
//            output_tensor.set<ValueType>({b, c, oh, ow}, sum > 0 ? sum : 0);
//          }
//        }
//      },
//      options);
//
//  output = output_tensor;
//}
//
//// NCHW -> NCHW only (Legacy version)
//template <typename ValueType>
//void Conv4D_LegacyRelu(const Tensor& input, const Tensor& kernel_,
//                       const Tensor& bias_, Tensor& output, size_t stride_,
//                       size_t pads_, size_t dilations_,
//                       ParBackend backend = ParBackend::kSeq) {
//  size_t batch_size = input.get_shape()[0];
//  size_t in_height = input.get_shape()[2];
//  size_t in_width = input.get_shape()[3];
//  size_t in_channels = input.get_shape()[1];
//
//  size_t kernel_height = kernel_.get_shape()[0];
//  size_t kernel_width = kernel_.get_shape()[1];
//  size_t kernel_in_channels = kernel_.get_shape()[2];
//  size_t kernel_out_channels = kernel_.get_shape()[3];
//
//  parallel::Options options;
//  options.backend = backend;
//
//  const auto& input_data = *input.as<ValueType>();
//  const auto& kernel_data = *kernel_.as<ValueType>();
//  const std::vector<ValueType>* bias_data = nullptr;
//  if (!bias_.empty()) {
//    bias_data = bias_.as<ValueType>();
//  }
//
//  size_t out_height = ComputeConvOutputDim(in_height, kernel_height, stride_,
//                                           pads_, dilations_);
//  size_t out_width =
//      ComputeConvOutputDim(in_width, kernel_width, stride_, pads_, dilations_);
//
//  const size_t input_channel_stride = in_height * in_width;
//  const size_t input_batch_stride = in_channels * input_channel_stride;
//  const size_t kernel_channel_stride = kernel_height * kernel_width;
//  const size_t kernel_output_stride =
//      kernel_in_channels * kernel_channel_stride;
//  const size_t output_channel_stride = out_height * out_width;
//  const size_t output_batch_stride = kernel_out_channels * output_channel_stride;
//  const size_t in_channels_per_group = in_channels;
//  const size_t out_channels_per_group = kernel_out_channels;
//  const bool collapsed_kernel = dilations_ == 0;
//
//  Shape output_shape({batch_size, kernel_out_channels, out_height, out_width});
//  std::vector<ValueType> flat_output(output_shape.count(), 0);
//
//  size_t total_work = batch_size * kernel_out_channels;
//
//  parallel::parallel_for(
//      total_work,
//      [&](size_t idx) {
//        size_t b = idx / kernel_out_channels;
//        size_t c = idx % kernel_out_channels;
//        size_t input_batch_base = b * input_batch_stride;
//        size_t output_base =
//            b * output_batch_stride + c * output_channel_stride;
//        size_t group = 0;
//        size_t group_start_channel = 0;
//        size_t group_end_channel = in_channels_per_group;
//        size_t kernel_c_base = c * kernel_output_stride;
//        auto bias_value = ValueType{};
//        if (bias_data != nullptr && c < bias_data->size()) {
//          bias_value = (*bias_data)[c];
//        }
//
//        for (size_t i = 0; i < out_height; i ++) {
//          std::ptrdiff_t input_h_base =
//              static_cast<std::ptrdiff_t>(i * stride_) -
//              static_cast<std::ptrdiff_t>(pads_);
//          for (size_t j = 0; j < out_width; j ++) {
//            ValueType value = bias_value;
//            std::ptrdiff_t input_w_base =
//                static_cast<std::ptrdiff_t>(j * stride_) -
//                static_cast<std::ptrdiff_t>(pads_);
//            size_t output_idx = output_base + i * out_width + j;
//            for (size_t ic = 0; ic < in_channels; ++ic) {
//              size_t kernel_ic = ic - group_start_channel;
//              size_t input_channel_base =
//                  input_batch_base + ic * input_channel_stride;
//              size_t kernel_ic_base =
//                  kernel_c_base + kernel_ic * kernel_channel_stride;
//              if (collapsed_kernel) {
//                if (input_h_base >= 0 &&
//                    input_h_base < static_cast<std::ptrdiff_t>(in_height) &&
//                    input_w_base >= 0 &&
//                    input_w_base < static_cast<std::ptrdiff_t>(in_width)) {
//                  size_t input_idx =
//                      input_channel_base +
//                      static_cast<size_t>(input_h_base) * in_width +
//                      static_cast<size_t>(input_w_base);
//                  size_t kernel_idx =
//                      kernel_ic_base + kernel_channel_stride - 1;
//                  value += input_data[input_idx] * kernel_data[kernel_idx];
//                }
//                continue;
//              }
//              for (size_t kh = 0; kh < kernel_height; ++kh) {
//                std::ptrdiff_t input_h =
//                    input_h_base + static_cast<std::ptrdiff_t>(kh * dilations_);
//                if (input_h < 0 ||
//                    input_h >= static_cast<std::ptrdiff_t>(in_height)) {
//                  continue;
//                }
//                size_t input_row_base = input_channel_base +
//                                        static_cast<size_t>(input_h) * in_width;
//                size_t kernel_row_base = kernel_ic_base + kh * kernel_width;
//
//                for (size_t kw = 0; kw < kernel_width; ++kw) {
//                  std::ptrdiff_t input_w =
//                      input_w_base +
//                      static_cast<std::ptrdiff_t>(kw * dilations_);
//                  if (input_w < 0 ||
//                      input_w >= static_cast<std::ptrdiff_t>(in_width)) {
//                    continue;
//                  }
//
//                  value += input_data[input_row_base +
//                                      static_cast<size_t>(input_w)] *
//                           kernel_data[kernel_row_base + kw];
//                }
//              }
//            }
//            flat_output[output_idx] = value > 0 ? value : 0;
//          }
//        }
//      },
//      options);
//
//  output = make_tensor<ValueType>(flat_output, output_shape);
//}
}  // namespace it_lab_ai
