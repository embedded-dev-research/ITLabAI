#include "layers/ConvLayer.hpp"

namespace itlab_2023 {

void ConvolutionalLayer::run(const Tensor& input, Tensor& output) {
  switch (input.get_type()) {
    case Type::kInt: {
      ConvImpl<int> used_impl(
          stride_, pads_, dilations_,
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
          input.get_shape()[input.get_shape().dims() - 1] *
              input.get_shape()[input.get_shape().dims() - 2],
          bias_.empty() ? std::vector<int>()
                        : std::vector<int>(bias_.begin(), bias_.end()));

      if (input.get_shape().dims() != 4) {
        throw std::out_of_range("Input must be 4-dimensional");
      }

      if (kernel_.get_shape().dims() == 2) {
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output = make_tensor<int>(
            used_impl.run(
                *input.as<int>(),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 2]) +
                    2 * static_cast<int>(pads_),
                *kernel_.as<int>(),
                kernel_.get_shape()[kernel_.get_shape().dims() - 1],
                (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                        dilations_ +
                    kernel_.get_shape()[kernel_.get_shape().dims() - 1],
                static_cast<int>(
                    ((1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                         dilations_ +
                     kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1) /
                    2)),
            sh);
      } else {
        size_t batch_size = input.get_shape()[0];
        size_t in_height = input.get_shape()[2];
        size_t in_width = input.get_shape()[3];
        size_t in_channels = input.get_shape()[1];

        size_t kernel_height = kernel_.get_shape()[0];
        size_t kernel_width = kernel_.get_shape()[1];
        size_t kernel_in_channels = kernel_.get_shape()[2];
        size_t kernel_out_channels = kernel_.get_shape()[3];

        std::vector<int> t = *input.as<int>();
        std::vector<std::vector<std::vector<std::vector<int>>>> input_tensor(
            batch_size,
            std::vector<std::vector<std::vector<int>>>(
                in_height, std::vector<std::vector<int>>(
                               in_width, std::vector<int>(in_channels, 1))));
        for (size_t index = 0; index < t.size(); ++index) {
          size_t n_index = index / (in_height * in_width * in_channels);
          size_t h_index = (index / (in_width * in_channels)) % in_height;
          size_t w_index = (index / in_channels) % in_width;
          size_t c_index = index % in_channels;
          input_tensor[n_index][h_index][w_index][c_index] = t[index];
        }

        std::vector<int> t1 = *kernel_.as<int>();
        std::vector<std::vector<std::vector<std::vector<int>>>> kernel(
            kernel_height,
            std::vector<std::vector<std::vector<int>>>(
                kernel_width, std::vector<std::vector<int>>(
                                  kernel_in_channels,
                                  std::vector<int>(kernel_out_channels, 1))));
        for (size_t index = 0; index < t1.size(); ++index) {
          size_t n_index =
              index / (kernel_width * kernel_in_channels * kernel_out_channels);
          size_t h_index =
              (index / (kernel_in_channels * kernel_out_channels)) %
              kernel_width;
          size_t w_index = (index / kernel_out_channels) % kernel_in_channels;
          size_t c_index = index % kernel_out_channels;
          kernel[n_index][h_index][w_index][c_index] = t1[index];
        }

        pads_ = (kernel_height * (1 + 2 * dilations_) - 1) / 2;

        std::vector<std::vector<std::vector<std::vector<int>>>> padded_input =
            input_tensor;
        if (pads_ > 0) {
          padded_input =
              std::vector<std::vector<std::vector<std::vector<int>>>>(
                  batch_size, std::vector<std::vector<std::vector<int>>>(
                                  in_height + 2 * pads_,
                                  std::vector<std::vector<int>>(
                                      in_width + 2 * pads_,
                                      std::vector<int>(in_channels, 0))));

          for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < in_height; ++h) {
              for (size_t w = 0; w < in_width; ++w) {
                for (size_t c = 0; c < in_channels; ++c) {
                  padded_input[b][h + pads_][w + pads_][c] =
                      input_tensor[b][h][w][c];
                }
              }
            }
          }
        }

        std::vector<std::vector<std::vector<std::vector<int>>>> dil_kernel =
            kernel;
        if (dilations_ > 0) {
          dil_kernel = std::vector<std::vector<std::vector<std::vector<int>>>>(
              kernel_height * (1 + 2 * dilations_),
              std::vector<std::vector<std::vector<int>>>(
                  kernel_width * (1 + 2 * dilations_),
                  std::vector<std::vector<int>>(
                      kernel_in_channels,
                      std::vector<int>(kernel_out_channels, 0))));

          for (size_t b = 0; b < kernel_out_channels; ++b) {
            for (size_t h = 0; h < kernel_height; ++h) {
              for (size_t w = 0; w < kernel_width; ++w) {
                for (size_t c = 0; c < kernel_in_channels; ++c) {
                  dil_kernel[(h * (1 + 2 * dilations_)) + dilations_]
                            [(w * (1 + 2 * dilations_)) + dilations_][c][b] =
                                kernel[h][w][c][b];
                }
              }
            }
          }
        }

        size_t crat = 0;
        if ((in_height + 2 * pads_ -
             ((kernel_height * (1 + 2 * dilations_)) - 1)) %
                stride_ !=
            0)
          crat = 1;

        size_t out_height = (in_height + 2 * pads_ -
                             ((kernel_height * (1 + 2 * dilations_)) - 1)) /
                                stride_ +
                            crat;

        crat = 0;

        if ((in_width + 2 * pads_ -
             ((kernel_width * (1 + 2 * dilations_)) - 1)) %
                stride_ !=
            0)
          crat = 1;

        size_t out_width = (in_width + 2 * pads_ -
                            ((kernel_width * (1 + 2 * dilations_)) - 1)) /
                               stride_ +
                           crat;

        std::vector<std::vector<std::vector<std::vector<int>>>> output_tensor(
            batch_size,
            std::vector<std::vector<std::vector<int>>>(
                out_height,
                std::vector<std::vector<int>>(
                    out_width, std::vector<int>(kernel_out_channels, 0))));
        size_t one_size = (kernel_height * (1 + 2 * dilations_) - 1) / 2;

        for (size_t b = 0; b < batch_size; ++b) {
          for (size_t c = 0; c < kernel_out_channels; ++c) {
            for (size_t i = 0; i < out_height; i += stride_) {
              for (size_t j = 0; j < out_width; j += stride_) {
                int value = 0;
                for (size_t ic = 0; ic < in_channels; ++ic) {
                  for (int h = (-1 * static_cast<int>(one_size));
                       h <= static_cast<int>(one_size); ++h) {
                    for (int w = (-1 * static_cast<int>(one_size));
                         w <= static_cast<int>(one_size); ++w) {
                      value += padded_input[b][i + one_size + h]
                                           [j + one_size + w][ic] *
                               dil_kernel[one_size + h][one_size + w][ic][c];
                    }
                  }
                }
                output_tensor[b][i][j][c] = value;
              }
            }
          }
        }

        Shape sh({batch_size, kernel_out_channels, out_height, out_width});
        std::vector<int> one_d_vector(batch_size * out_height * out_width *
                                      kernel_out_channels);
        size_t index_1d = 0;
        for (size_t i = 0; i < batch_size; ++i) {
          for (size_t l = 0; l < kernel_out_channels; ++l) {
            for (size_t j = 0; j < out_height; ++j) {
              for (size_t k = 0; k < out_width; ++k) {
                one_d_vector[index_1d++] = output_tensor[i][j][k][l];
              }
            }
          }
        }
        output = make_tensor<int>(one_d_vector, sh);
      }
      break;
    }
    case Type::kFloat: {
      ConvImpl<float> used_impl(
          stride_, pads_, dilations_,
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 1]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 2]),
          static_cast<int>(input.get_shape()[input.get_shape().dims() - 3]),
          input.get_shape()[input.get_shape().dims() - 1] *
              input.get_shape()[input.get_shape().dims() - 2],
          bias_.empty() ? std::vector<float>()
                        : std::vector<float>(bias_.begin(), bias_.end()));

      if (input.get_shape().dims() != 4) {
        throw std::out_of_range("Input must be 4-dimensional");
      }

      if (kernel_.get_shape().dims() == 2) {
        auto sizeforshape = static_cast<size_t>(
            ((static_cast<int>(
                  input.get_shape()[input.get_shape().dims() - 1]) -
              1 -
              static_cast<int>(
                  (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                      dilations_ +
                  kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1)) /
             static_cast<int>(stride_)) +
            1);

        Shape sh({1, 3, sizeforshape, sizeforshape});
        output = make_tensor<float>(
            used_impl.run(
                *input.as<float>(),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 1]) +
                    2 * static_cast<int>(pads_),
                static_cast<int>(
                    input.get_shape()[input.get_shape().dims() - 2]) +
                    2 * static_cast<int>(pads_),
                *kernel_.as<float>(),
                kernel_.get_shape()[kernel_.get_shape().dims() - 1],
                (1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                        dilations_ +
                    kernel_.get_shape()[kernel_.get_shape().dims() - 1],
                static_cast<int>(
                    ((1 + kernel_.get_shape()[kernel_.get_shape().dims() - 1]) *
                         dilations_ +
                     kernel_.get_shape()[kernel_.get_shape().dims() - 1] - 1) /
                    2)),
            sh);
      } else {
        size_t batch_size = input.get_shape()[0];
        size_t in_height = input.get_shape()[2];
        size_t in_width = input.get_shape()[3];
        size_t in_channels = input.get_shape()[1];

        size_t kernel_height = kernel_.get_shape()[0];
        size_t kernel_width = kernel_.get_shape()[1];
        size_t kernel_in_channels = kernel_.get_shape()[2];
        size_t kernel_out_channels = kernel_.get_shape()[3];

        std::vector<float> t = *input.as<float>();
        std::vector<std::vector<std::vector<std::vector<float>>>> input_tensor(
            batch_size,
            std::vector<std::vector<std::vector<float>>>(
                in_height, std::vector<std::vector<float>>(
                               in_width, std::vector<float>(in_channels, 1))));
        for (size_t index = 0; index < t.size(); ++index) {
          size_t n_index = index / (in_height * in_width * in_channels);
          size_t h_index = (index / (in_width * in_channels)) % in_height;
          size_t w_index = (index / in_channels) % in_width;
          size_t c_index = index % in_channels;
          input_tensor[n_index][h_index][w_index][c_index] = t[index];
        }

        std::vector<float> t1 = *kernel_.as<float>();
        std::vector<std::vector<std::vector<std::vector<float>>>> kernel(
            kernel_height,
            std::vector<std::vector<std::vector<float>>>(
                kernel_width, std::vector<std::vector<float>>(
                                  kernel_in_channels,
                                  std::vector<float>(kernel_out_channels, 1))));
        for (size_t index = 0; index < t1.size(); ++index) {
          size_t n_index =
              index / (kernel_width * kernel_in_channels * kernel_out_channels);
          size_t h_index =
              (index / (kernel_in_channels * kernel_out_channels)) %
              kernel_width;
          size_t w_index = (index / kernel_out_channels) % kernel_in_channels;
          size_t c_index = index % kernel_out_channels;
          kernel[n_index][h_index][w_index][c_index] = t1[index];
        }

        pads_ = (kernel_height * (1 + 2 * dilations_) - 1) / 2;

        std::vector<std::vector<std::vector<std::vector<float>>>> padded_input =
            input_tensor;
        if (pads_ > 0) {
          padded_input =
              std::vector<std::vector<std::vector<std::vector<float>>>>(
                  batch_size, std::vector<std::vector<std::vector<float>>>(
                                  in_height + 2 * pads_,
                                  std::vector<std::vector<float>>(
                                      in_width + 2 * pads_,
                                      std::vector<float>(in_channels, 0))));

          for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < in_height; ++h) {
              for (size_t w = 0; w < in_width; ++w) {
                for (size_t c = 0; c < in_channels; ++c) {
                  padded_input[b][h + pads_][w + pads_][c] =
                      input_tensor[b][h][w][c];
                }
              }
            }
          }
        }

        std::vector<std::vector<std::vector<std::vector<float>>>> dil_kernel =
            kernel;
        if (dilations_ > 0) {
          dil_kernel =
              std::vector<std::vector<std::vector<std::vector<float>>>>(
                  kernel_height * (1 + 2 * dilations_),
                  std::vector<std::vector<std::vector<float>>>(
                      kernel_width * (1 + 2 * dilations_),
                      std::vector<std::vector<float>>(
                          kernel_in_channels,
                          std::vector<float>(kernel_out_channels, 0))));

          for (size_t b = 0; b < kernel_out_channels; ++b) {
            for (size_t h = 0; h < kernel_height; ++h) {
              for (size_t w = 0; w < kernel_width; ++w) {
                for (size_t c = 0; c < kernel_in_channels; ++c) {
                  dil_kernel[(h * (1 + 2 * dilations_)) + dilations_]
                            [(w * (1 + 2 * dilations_)) + dilations_][c][b] =
                                kernel[h][w][c][b];
                }
              }
            }
          }
        }

        size_t crat = 0;
        if ((in_height + 2 * pads_ -
             ((kernel_height * (1 + 2 * dilations_)) - 1)) %
                stride_ !=
            0)
          crat = 1;

        size_t out_height = (in_height + 2 * pads_ -
                             ((kernel_height * (1 + 2 * dilations_)) - 1)) /
                                stride_ +
                            crat;

        crat = 0;

        if ((in_width + 2 * pads_ -
             ((kernel_width * (1 + 2 * dilations_)) - 1)) %
                stride_ !=
            0)
          crat = 1;

        size_t out_width = (in_width + 2 * pads_ -
                            ((kernel_width * (1 + 2 * dilations_)) - 1)) /
                               stride_ +
                           crat;

        std::vector<std::vector<std::vector<std::vector<float>>>> output_tensor(
            batch_size,
            std::vector<std::vector<std::vector<float>>>(
                out_height,
                std::vector<std::vector<float>>(
                    out_width, std::vector<float>(kernel_out_channels, 0))));
        size_t one_size = (kernel_height * (1 + 2 * dilations_) - 1) / 2;

        for (size_t b = 0; b < batch_size; ++b) {
          for (size_t c = 0; c < kernel_out_channels; ++c) {
            for (size_t i = 0; i < out_height; i += stride_) {
              for (size_t j = 0; j < out_width; j += stride_) {
                float value = 0;
                for (size_t ic = 0; ic < in_channels; ++ic) {
                  for (int h = (-1 * static_cast<int>(one_size));
                       h <= static_cast<int>(one_size); ++h) {
                    for (int w = (-1 * static_cast<int>(one_size));
                         w <= static_cast<int>(one_size); ++w) {
                      value += padded_input[b][i + one_size + h]
                                           [j + one_size + w][ic] *
                               dil_kernel[one_size + h][one_size + w][ic][c];
                    }
                  }
                }
                output_tensor[b][i][j][c] = value;
              }
            }
          }
        }

        Shape sh({batch_size, kernel_out_channels, out_height, out_width});
        std::vector<float> one_d_vector(batch_size * out_height * out_width *
                                        kernel_out_channels);
        size_t index_1d = 0;
        for (size_t i = 0; i < batch_size; ++i) {
          for (size_t l = 0; l < kernel_out_channels; ++l) {
            for (size_t j = 0; j < out_height; ++j) {
              for (size_t k = 0; k < out_width; ++k) {
                one_d_vector[index_1d++] = output_tensor[i][j][k][l];
              }
            }
          }
        }
        output = make_tensor<float>(one_d_vector, sh);
      }
      break;
    }
    default: {
      throw std::runtime_error("Unsupported tensor type");
    }
  }
}

}  // namespace itlab_2023