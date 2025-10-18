#pragma once
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class ConvolutionalLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  Tensor kernel_;
  Tensor bias_;
  size_t group_;
  ImplType implType_;
  bool useLegacyImpl_;

 public:
  ConvolutionalLayer() : Layer(kConvolution) {
    stride_ = 0;
    pads_ = 0;
    dilations_ = 0;
    implType_ = kDefault;
  }
  ConvolutionalLayer(size_t step, size_t pads, size_t dilations,
                     const Tensor& kernel, const Tensor& bias = Tensor(),
                     ImplType implType = kDefault, size_t group = 1,
                     bool useLegacyImpl = false)
      : Layer(kConvolution) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    kernel_ = kernel;
    bias_ = bias;
    implType_ = implType;
    useLegacyImpl_ = useLegacyImpl;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return kernel_; }
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

  std::vector<ValueType> run(
      const std::vector<ValueType>& input) const override {
    return input;
  }

  std::vector<ValueType> run(std::vector<ValueType> startmatrix, int new_rows,
                             int new_cols, std::vector<ValueType> startkernel,
                             size_t start_kernel_size, size_t kernel_size,
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
            auto kercol = static_cast<size_t>(coloms / input_width_ + 1);
            color +=
                matrix[(i + coloms + str) * input_flow_ + x] *
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
            size_t dilations_) {
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

  size_t out_height =
      (in_height + 2 * pads_ - dilations_ * (kernel_height - 1) - 1) / stride_ +
      1;
  size_t out_width =
      (in_width + 2 * pads_ - dilations_ * (kernel_width - 1) - 1) / stride_ +
      1;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          in_height + 2 * pads_,
          std::vector<std::vector<ValueType>>(
              in_width + 2 * pads_, std::vector<ValueType>(in_channels, 0))));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < in_height; ++h) {
      for (size_t w = 0; w < in_width; ++w) {
        for (size_t c = 0; c < in_channels; ++c) {
          padded_input[b][h + pads_][w + pads_][c] =
              input.get<ValueType>({b, c, h, w});
        }
      }
    }
  }

  size_t dilated_kernel_height = (kernel_height - 1) * dilations_ + 1;
  size_t dilated_kernel_width = (kernel_width - 1) * dilations_ + 1;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel(
      out_channels, std::vector<std::vector<std::vector<ValueType>>>(
                        kernel_in_channels,
                        std::vector<std::vector<ValueType>>(
                            dilated_kernel_height,
                            std::vector<ValueType>(dilated_kernel_width, 0))));

  for (size_t oc = 0; oc < out_channels; ++oc) {
    for (size_t ic = 0; ic < kernel_in_channels; ++ic) {
      for (size_t kh = 0; kh < kernel_height; ++kh) {
        for (size_t kw = 0; kw < kernel_width; ++kw) {
          dil_kernel[oc][ic][kh * dilations_][kw * dilations_] =
              kernel_.get<ValueType>({oc, ic, kh, kw});
        }
      }
    }
  }

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size,
      std::vector<std::vector<std::vector<ValueType>>>(
          out_channels, std::vector<std::vector<ValueType>>(
                            out_height, std::vector<ValueType>(out_width, 0))));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oc = 0; oc < out_channels; ++oc) {
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
    }
  }

  Shape output_shape({batch_size, out_channels, out_height, out_width});
  std::vector<ValueType> flat_output(batch_size * out_channels * out_height *
                                     out_width);

  size_t index = 0;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oc = 0; oc < out_channels; ++oc) {
      for (size_t h = 0; h < out_height; ++h) {
        for (size_t w = 0; w < out_width; ++w) {
          flat_output[index++] = output_tensor[b][oc][h][w];
        }
      }
    }
  }

  output = make_tensor<ValueType>(flat_output, output_shape);
}

// NCHW -> NCHW only
template <typename ValueType>
void Conv4DSTL(const Tensor& input, const Tensor& kernel_, const Tensor& bias_,
               Tensor& output, size_t stride_, size_t pads_, size_t group_,
               size_t dilations_) {
  size_t batch_size = input.get_shape()[0];
  size_t in_channels = input.get_shape()[1];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];

  size_t kernel_out_channels = kernel_.get_shape()[0];
  size_t kernel_in_channels = kernel_.get_shape()[1];
  size_t kernel_height = kernel_.get_shape()[2];
  size_t kernel_width = kernel_.get_shape()[3];

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  size_t chunk_size = batch_size / num_threads;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input =
      std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
          batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                          in_height + 2 * pads_,
                          std::vector<std::vector<ValueType>>(
                              in_width + 2 * pads_,
                              std::vector<ValueType>(in_channels, 0))));
  auto pad_input = [&](size_t start_b, size_t end_b) {
    for (size_t b = start_b; b < end_b; ++b) {
      for (size_t h = 0; h < in_height; ++h) {
        for (size_t w = 0; w < in_width; ++w) {
          for (size_t c = 0; c < in_channels; ++c) {
            padded_input[b][h + pads_][w + pads_][c] =
                input.get<ValueType>({b, c, h, w});
          }
        }
      }
    }
  };

  for (unsigned i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? batch_size : start + chunk_size;
    threads.emplace_back(pad_input, start, end);
  }
  for (auto& t : threads) t.join();
  threads.clear();
  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel =
      std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
          kernel_height * dilations_ + 1 - dilations_,
          std::vector<std::vector<std::vector<ValueType>>>(
              kernel_width * dilations_ + 1 - dilations_,
              std::vector<std::vector<ValueType>>(
                  kernel_in_channels,
                  std::vector<ValueType>(kernel_out_channels, 0))));

  auto dilate_kernel = [&](size_t start_oc, size_t end_oc) {
    for (size_t oc = start_oc; oc < end_oc; ++oc) {
      for (size_t h = 0; h < kernel_height; ++h) {
        for (size_t w = 0; w < kernel_width; ++w) {
          for (size_t ic = 0; ic < kernel_in_channels; ++ic) {
            dil_kernel[h * dilations_][w * dilations_][ic][oc] =
                kernel_.get<ValueType>({oc, ic, h, w});
          }
        }
      }
    }
  };

  chunk_size = kernel_out_channels / num_threads;
  for (unsigned i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end =
        (i == num_threads - 1) ? kernel_out_channels : start + chunk_size;
    threads.emplace_back(dilate_kernel, start, end);
  }
  for (auto& t : threads) t.join();
  threads.clear();

  size_t crat = 0;
  if ((in_height + 2 * pads_ - dilations_ * (kernel_height - 1)) % stride_ != 0)
    crat = 1;

  size_t out_height =
      (in_height + 2 * pads_ - dilations_ * (kernel_height - 1)) / stride_ +
      crat;

  crat = 0;
  if ((in_width + 2 * pads_ - dilations_ * (kernel_width - 1)) % stride_ != 0)
    crat = 1;

  size_t out_width =
      (in_width + 2 * pads_ - dilations_ * (kernel_width - 1)) / stride_ + crat;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                      kernel_out_channels,
                      std::vector<std::vector<ValueType>>(
                          out_height, std::vector<ValueType>(out_width, 0))));

  auto compute_conv = [&](size_t start_oc, size_t end_oc) {
    size_t dilated_kernel_height = kernel_height * dilations_ + 1 - dilations_;
    size_t dilated_kernel_width = kernel_width * dilations_ + 1 - dilations_;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t oc = start_oc; oc < end_oc; ++oc) {
        for (size_t oh = 0; oh < out_height; oh++) {
          for (size_t ow = 0; ow < out_width; ow++) {
            ValueType value = 0;

            size_t group =
                (group_ > 1) ? oc / (kernel_out_channels / group_) : 0;
            size_t group_start_channel = group * (in_channels / group_);
            size_t group_end_channel = (group + 1) * (in_channels / group_);

            for (size_t ic = group_start_channel; ic < group_end_channel;
                 ++ic) {
              size_t kernel_ic = ic - group_start_channel;

              for (size_t kh = 0; kh < dilated_kernel_height; ++kh) {
                for (size_t kw = 0; kw < dilated_kernel_width; ++kw) {
                  size_t h_index = oh * stride_ + kh;
                  size_t w_index = ow * stride_ + kw;

                  if (h_index < padded_input[b].size() &&
                      w_index < padded_input[b][h_index].size()) {
                    value += padded_input[b][h_index][w_index][ic] *
                             dil_kernel[kh][kw][kernel_ic][oc];
                  }
                }
              }
            }

            if (!bias_.empty()) {
              output_tensor[b][oc][oh][ow] =
                  value + (*bias_.as<ValueType>())[oc];
            } else {
              output_tensor[b][oc][oh][ow] = value;
            }
          }
        }
      }
    }
  };

  chunk_size = kernel_out_channels / num_threads;
  for (unsigned i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end =
        (i == num_threads - 1) ? kernel_out_channels : start + chunk_size;
    threads.emplace_back(compute_conv, start, end);
  }
  for (auto& t : threads) t.join();
  threads.clear();

  Shape sh({batch_size, kernel_out_channels, out_height, out_width});
  std::vector<ValueType> one_d_vector(batch_size * out_height * out_width *
                                      kernel_out_channels);

  auto flatten_output = [&](size_t start_b, size_t end_b) {
    size_t index_1d = start_b * kernel_out_channels * out_height * out_width;
    for (size_t i = start_b; i < end_b; ++i) {
      for (size_t l = 0; l < kernel_out_channels; ++l) {
        for (size_t j = 0; j < out_height; ++j) {
          for (size_t k = 0; k < out_width; ++k) {
            one_d_vector[index_1d++] = output_tensor[i][l][j][k];
          }
        }
      }
    }
  };

  chunk_size = batch_size / num_threads;
  for (unsigned i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? batch_size : start + chunk_size;
    threads.emplace_back(flatten_output, start, end);
  }
  for (auto& t : threads) t.join();

  output = make_tensor<ValueType>(one_d_vector, sh);
}

template <typename ValueType>
void DepthwiseConv4D(const Tensor& input, const Tensor& kernel_,
                     const Tensor& bias_, Tensor& output, size_t stride_,
                     size_t pads_, size_t dilations_) {
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

  size_t out_height =
      (in_height + 2 * pads_ - dilations_ * (kernel_height - 1) - 1) / stride_ +
      1;
  size_t out_width =
      (in_width + 2 * pads_ - dilations_ * (kernel_width - 1) - 1) / stride_ +
      1;

  Tensor output_tensor(Shape({batch_size, channels, out_height, out_width}),
                       input.get_type());

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < out_height; ++oh) {
        for (size_t ow = 0; ow < out_width; ++ow) {
          ValueType sum = 0;

          for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
              size_t ih = oh * stride_ + kh * dilations_ - pads_;
              size_t iw = ow * stride_ + kw * dilations_ - pads_;

              if (ih < in_height && iw < in_width) {
                auto input_val = input.get<ValueType>({b, c, ih, iw});
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
    }
  }

  output = output_tensor;
}

// NCHW -> NCHW only
template <typename ValueType>
void Conv4D_Legacy(const Tensor& input, const Tensor& kernel_,
                   const Tensor& bias_, Tensor& output, size_t stride_,
                   size_t pads_, size_t dilations_) {
  size_t batch_size = input.get_shape()[0];
  size_t in_height = input.get_shape()[2];
  size_t in_width = input.get_shape()[3];
  size_t in_channels = input.get_shape()[1];

  size_t kernel_height = kernel_.get_shape()[0];
  size_t kernel_width = kernel_.get_shape()[1];
  size_t kernel_in_channels = kernel_.get_shape()[2];
  size_t kernel_out_channels = kernel_.get_shape()[3];

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> padded_input =
      std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
          batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                          in_height + 2 * pads_,
                          std::vector<std::vector<ValueType>>(
                              in_width + 2 * pads_,
                              std::vector<ValueType>(in_channels, 0))));
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t h = 0; h < in_height; ++h) {
      for (size_t w = 0; w < in_width; ++w) {
        for (size_t c = 0; c < in_channels; ++c) {
          padded_input[b][h + pads_][w + pads_][c] =
              input.get<ValueType>({b, c, h, w});
        }
      }
    }
  }
  std::vector<std::vector<std::vector<std::vector<ValueType>>>> dil_kernel =
      std::vector<std::vector<std::vector<std::vector<ValueType>>>>(
          kernel_height * dilations_ + 1 - dilations_,
          std::vector<std::vector<std::vector<ValueType>>>(
              kernel_width * dilations_ + 1 - dilations_,
              std::vector<std::vector<ValueType>>(
                  kernel_in_channels,
                  std::vector<ValueType>(kernel_out_channels, 0))));
  for (size_t b = 0; b < kernel_out_channels; ++b) {
    for (size_t h = 0; h < kernel_height; ++h) {
      for (size_t w = 0; w < kernel_width; ++w) {
        for (size_t c = 0; c < kernel_in_channels; ++c) {
          dil_kernel[h * dilations_][w * dilations_][c][b] =
              kernel_.get<ValueType>({h, w, c, b});
        }
      }
    }
  }

  size_t crat = 0;
  if ((in_height + 2 * pads_ - dilations_ * (kernel_height - 1)) % stride_ != 0)
    crat = 1;

  size_t out_height =
      (in_height + 2 * pads_ - dilations_ * (kernel_height - 1)) / stride_ +
      crat;

  crat = 0;
  if ((in_width + 2 * pads_ - dilations_ * (kernel_width - 1)) % stride_ != 0)
    crat = 1;

  size_t out_width =
      (in_width + 2 * pads_ - dilations_ * (kernel_width - 1)) / stride_ + crat;

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> output_tensor(
      batch_size, std::vector<std::vector<std::vector<ValueType>>>(
                      kernel_out_channels,
                      std::vector<std::vector<ValueType>>(
                          out_height, std::vector<ValueType>(out_width, 0))));
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < kernel_out_channels; ++c) {
      for (size_t i = 0; i < out_height; i += stride_) {
        for (size_t j = 0; j < out_width; j += stride_) {
          ValueType value = 0;
          for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t h = 0; h < kernel_height * dilations_ + 1 - dilations_;
                 ++h) {
              for (size_t w = 0; w < kernel_width * dilations_ + 1 - dilations_;
                   ++w) {
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
    }
  }

  Shape sh({batch_size, kernel_out_channels, out_height, out_width});
  std::vector<ValueType> one_d_vector(batch_size * out_height * out_width *
                                      kernel_out_channels);
  size_t index_1d = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t l = 0; l < kernel_out_channels; ++l) {
      for (size_t j = 0; j < out_height; ++j) {
        for (size_t k = 0; k < out_width; ++k) {
          one_d_vector[index_1d++] = output_tensor[i][l][j][k];
        }
      }
    }
  }
  output = make_tensor<ValueType>(one_d_vector, sh);
}
}  // namespace it_lab_ai
