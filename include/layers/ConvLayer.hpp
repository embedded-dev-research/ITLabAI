#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

inline size_t ComputeConvOutputDim(size_t input_size, size_t kernel_size,
                                   size_t stride, size_t padding,
                                   size_t dilation) {
  const size_t effective_kernel = dilation * (kernel_size - 1) + 1;
  if (stride == 0 || input_size + 2 * padding < effective_kernel) {
    return 0;
  }
  return (input_size + 2 * padding - effective_kernel) / stride + 1;
}

class ConvolutionalLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  std::shared_ptr<Tensor> kernel_;
  std::shared_ptr<Tensor> bias_;
  size_t group_;
  bool useLegacyImpl_;

 public:
  ConvolutionalLayer() : Layer(kConvolution), kernel_(nullptr), bias_(nullptr) {
    stride_ = 0;
    pads_ = 0;
    dilations_ = 0;
  }
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations,
                     const Tensor& kernel, const Tensor& bias = Tensor(),
                     size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvolution),
        kernel_(std::make_shared<Tensor>(kernel)),
        bias_(std::make_shared<Tensor>(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations,
                     std::shared_ptr<Tensor> kernel,
                     std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(),
                     size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvolution),
        kernel_(std::move(kernel)),
        bias_(std::move(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }

  [[nodiscard]] std::vector<size_t> getNumericParams() const {
    std::vector<size_t> res = {stride_, pads_, dilations_, group_};
    return res;
  }

  [[nodiscard]] std::vector<std::shared_ptr<Tensor>> getTensorParams() {
    std::vector<std::shared_ptr<Tensor>> res = {kernel_, bias_};
    return res;
  }

  [[nodiscard]] bool getLegacyImplBool() const {
    return useLegacyImpl_;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
           const RuntimeOptions& options) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    return *kernel_;
  }
#endif
};

template <typename ValueType>
class ConvImpl : public LayerImpl<ValueType> {
 private:
  int input_width_;
  int input_height_;
  int input_flow_;
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  size_t input_size_;
  std::vector<ValueType> bias_;

 public:
  ConvImpl() = delete;
  ConvImpl(size_t stride, size_t pads, size_t dilations, int input_width,
           int input_height, int input_flow, size_t input_size,
           const std::vector<ValueType>& bias)
      : input_width_(input_width),
        input_height_(input_height),
        input_flow_(input_flow),
        stride_(stride),
        pads_(pads),
        dilations_(dilations),
        input_size_(input_size),
        bias_(bias) {}

  ConvImpl(const ConvImpl& c) = default;

  [[nodiscard]] std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override {
    return input;
  }

  [[nodiscard]] std::vector<ValueType> run(std::vector<ValueType> startmatrix,
                                           int new_rows, int new_cols,
                                           std::vector<ValueType> startkernel,
                                           size_t start_kernel_size,
                                           size_t kernel_size,
                                           int center_distance) const {
    std::vector<ValueType> matrix(new_rows * new_cols * input_flow_, 0);
    for (int i = 0; i < input_height_; ++i) {
      for (int j = 0; j < input_width_; ++j) {
        for (int f = 0; f < input_flow_; ++f) {
          matrix[((i + pads_) * new_cols + j + pads_) * input_flow_ + f] =
              startmatrix[(i * input_width_ + j) * input_flow_ + f];
        }
      }
    }

    std::vector<ValueType> kernel(kernel_size * kernel_size, 0);
    for (int i = 0; i < static_cast<int>(start_kernel_size); ++i) {
      for (int j = 0; j < static_cast<int>(start_kernel_size); ++j) {
        kernel[(dilations_ + i) * static_cast<int>(kernel_size) + j +
               (j + 1) * dilations_] =
            startkernel[i * static_cast<int>(start_kernel_size) + j];
      }
    }

    std::vector<ValueType> outputvec;
    for (int i = input_width_ + center_distance;
         i < static_cast<int>(input_size_); i += static_cast<int>(stride_)) {
      for (int x = 0; x < input_flow_; ++x) {
        ValueType color = 0;
        for (int coloms = -input_width_; coloms < input_width_ + 1;
             coloms += input_width_) {
          for (int str = -1; str < 2; ++str) {
            if (input_width_ == 0) {
              throw std::out_of_range("Input = 0");
            }
            int kercol_index = coloms / input_width_ + 1;
            if (kercol_index < 0) {
              throw std::out_of_range("Kernel column index is negative");
            }
            auto kercol = static_cast<size_t>(kercol_index);
            color +=
                matrix.at((i + coloms + str) * input_flow_ + x) *
                kernel[kercol * kernel_size + static_cast<size_t>(str + 1)];
          }
        }
        if (!bias_.empty() && static_cast<size_t>(x) < bias_.size()) {
          color += bias_[x];
        }
        outputvec.push_back(color);
      }
      if ((i + center_distance + 1) % input_width_ == 0) {
        if (i + input_width_ + center_distance * 2 ==
            static_cast<int>(input_size_)) {
          i += input_width_ + center_distance * 2 + 1;
        } else {
          i += input_width_ * (static_cast<int>(stride_) - 1) +
               (3 - static_cast<int>(stride_));
        }
      }
    }
    return outputvec;
  }
};

// NCHW -> NCHW only
template <typename ValueType>
void Conv4D(const Tensor& input, const Tensor& kernel_, const Tensor& bias_,
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

  parallel::parallel_for(batch_size, [&](size_t b) {
    for (size_t h = 0; h < in_height; ++h) {
      for (size_t w = 0; w < in_width; ++w) {
        for (size_t c = 0; c < in_channels; ++c) {
          padded_input[b][h + pads_][w + pads_][c] =
              input.get<ValueType>({b, c, h, w});
        }
      }
    }
  }, options);

  size_t dilated_kernel_height = (kernel_height - 1) * dilations_ + 1;
  size_t dilated_kernel_width = (kernel_width - 1) * dilations_ + 1;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel(
      out_channels, std::vector<std::vector<std::vector<ValueType>>>(
                        kernel_in_channels,
                        std::vector<std::vector<ValueType>>(
                            dilated_kernel_height,
                            std::vector<ValueType>(dilated_kernel_width, 0))));

  parallel::parallel_for(out_channels, [&](size_t oc) {
    for (size_t ic = 0; ic < kernel_in_channels; ++ic) {
      for (size_t kh = 0; kh < kernel_height; ++kh) {
        for (size_t kw = 0; kw < kernel_width; ++kw) {
          dil_kernel[oc][ic][kh * dilations_][kw * dilations_] =
              kernel_.get<ValueType>({oc, ic, kh, kw});
        }
      }
    }
  }, options);

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          out_channels, std::vector<std::vector<ValueType>>(
                            out_height, std::vector<ValueType>(out_width, 0))));

  size_t total_work = batch_size * out_channels;
  parallel::parallel_for(total_work, [&](size_t idx) {
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

        for (size_t ic = group_start_channel; ic < group_end_channel; ++ic) {
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

        output_tensor[b][oc][oh][ow] = value;
      }
    }
  }, options);

  Shape output_shape({batch_size, out_channels, out_height, out_width});
  std::vector<ValueType> flat_output(batch_size * out_channels * out_height *
                                     out_width);

  parallel::parallel_for(batch_size, [&](size_t b) {
    size_t base_idx = b * out_channels * out_height * out_width;
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t h = 0; h < out_height; ++h) {
        for (size_t w = 0; w < out_width; ++w) {
          flat_output[base_idx++] = output_tensor[b][oc][h][w];
        }
      }
    }
  }, options);

  output = make_tensor<ValueType>(flat_output, output_shape);
}

template <typename ValueType>
void DepthwiseConv4D(const Tensor& input, const Tensor& kernel_,
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

  parallel::parallel_for(total_work, [&](size_t idx) {
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

        output_tensor.set<ValueType>({b, c, oh, ow}, sum);
      }
    }
  }, options);

  output = output_tensor;
}

// NCHW -> NCHW only (Legacy version)
template <typename ValueType>
void Conv4D_Legacy(const Tensor& input, const Tensor& kernel_,
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

  parallel::parallel_for(batch_size, [&](size_t b) {
    for (size_t h = 0; h < in_height; ++h) {
      for (size_t w = 0; w < in_width; ++w) {
        for (size_t c = 0; c < in_channels; ++c) {
          padded_input[b][h + pads_][w + pads_][c] =
              input.get<ValueType>({b, c, h, w});
        }
      }
    }
  }, options);

  size_t dilated_kernel_height = kernel_height * dilations_ + 1 - dilations_;
  size_t dilated_kernel_width = kernel_width * dilations_ + 1 - dilations_;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel(
      dilated_kernel_height,
      std::vector<std::vector<std::vector<ValueType>>>(
          dilated_kernel_width,
          std::vector<std::vector<ValueType>>(
              kernel_in_channels,
              std::vector<ValueType>(kernel_out_channels, 0))));

  parallel::parallel_for(kernel_out_channels, [&](size_t b) {
    for (size_t h = 0; h < kernel_height; ++h) {
      for (size_t w = 0; w < kernel_width; ++w) {
        for (size_t c = 0; c < kernel_in_channels; ++c) {
          dil_kernel[h * dilations_][w * dilations_][c][b] =
              kernel_.get<ValueType>({h, w, c, b});
        }
      }
    }
  }, options);

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

  parallel::parallel_for(total_work, [&](size_t idx) {
    size_t b = idx / kernel_out_channels;
    size_t c = idx % kernel_out_channels;

    for (size_t i = 0; i < out_height; i += stride_) {
      for (size_t j = 0; j < out_width; j += stride_) {
        ValueType value = 0;
        for (size_t ic = 0; ic < in_channels; ++ic) {
          for (size_t h = 0; h < dilated_kernel_height; ++h) {
            for (size_t w = 0; w < dilated_kernel_width; ++w) {
              value +=
                  padded_input[b][i + h][j + w][ic] * dil_kernel[h][w][ic][c];
            }
          }
        }
        if (!bias_.empty()) {
          output_tensor[b][c][i][j] = value + (*bias_.as<ValueType>())[c];
        } else {
          output_tensor[b][c][i][j] = value;
        }
      }
    }
  }, options);

  Shape sh({batch_size, kernel_out_channels, out_height, out_width});
  std::vector<ValueType> one_d_vector(batch_size * out_height * out_width *
                                      kernel_out_channels);

  parallel::parallel_for(batch_size, [&](size_t i) {
    size_t base_idx = i * kernel_out_channels * out_height * out_width;
    for (size_t l = 0; l < kernel_out_channels; ++l) {
      for (size_t j = 0; j < out_height; ++j) {
        for (size_t k = 0; k < out_width; ++k) {
          one_d_vector[base_idx++] = output_tensor[i][l][j][k];
        }
      }
    }
  }, options);

  output = make_tensor<ValueType>(one_d_vector, sh);
}

}  // namespace it_lab_ai
