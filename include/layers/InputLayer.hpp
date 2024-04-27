#pragma once
#include <algorithm>
#include <cmath>

#include "layers/Layer.hpp"

enum LayInOut {
  NCHW,  // 0
  NHWC   // 1
};

class InputLayer : public Layer {
 private:
  LayInOut layin_;
  LayInOut layout_;
  int mean_;
  int std_;

 public:
  InputLayer() = default;
  InputLayer(LayInOut layin, LayInOut layout, int mean, int std) {
    layin_ = layin;
    layout_ = layout;
    mean_ = mean;
    std_ = std;
  }  // layout = NCHW(0), NHWC(1)
  void run(Tensor& input, Tensor& output) const {
    switch (input.get_type()) {
      case Type::kInt: {
        std::vector<int> in = *input.as<int>();
        for (int& re : in) {
          re = static_cast<int>((re - mean_) / std_);
        }
        Shape sh(input.get_shape());
        if (layin_ == NCHW && layout_ == NHWC) {
          int N = static_cast<int>(sh[0]);
          int C = static_cast<int>(sh[1]);
          int H = static_cast<int>(sh[2]);
          int W = static_cast<int>(sh[3]);
          std::vector<int> res(N * H * W * C);
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                  int nchw_index = n * C * H * W + c * H * W + h * W + w;
                  int nhwc_index = n * H * W * C + h * W * C + w * C + c;
                  res[nhwc_index] = in[nchw_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(N),
                     static_cast<unsigned long long>(H),
                     static_cast<unsigned long long>(W),
                     static_cast<unsigned long long>(C)});
          output = make_tensor<int>(res, sh1);
          break;
        }
        if (layin_ == NHWC && layout_ == NCHW) {
          int N = static_cast<int> (sh[0]);
          int C = static_cast<int> (sh[3]);
          int H = static_cast<int> (sh[1]);
          int W = static_cast<int> (sh[2]);
          std::vector<int> res(N * C * H * W);
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                  int nhwc_index = n * H * W * C + h * W * C + w * C + c;
                  int nchw_index = n * C * H * W + c * H * W + h * W + w;
                  res[nchw_index] = in[nhwc_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(N),
                     static_cast<unsigned long long>(C),
                     static_cast<unsigned long long>(H),
                     static_cast<unsigned long long>(W)});
          output = make_tensor<int>(res, sh1);
          break;
        }
        output = make_tensor<int>(in, sh);
        break;
      }
      case Type::kFloat: {
        std::vector<float> in = *input.as<float>();
        for (float& re : in) {
          re = static_cast<float>((re - mean_) / std_);
        }
        Shape sh(input.get_shape());
        if (layin_ == NCHW && layout_ == NHWC) {
          int N = static_cast<int>(sh[0]);
          int C = static_cast<int>(sh[1]);
          int H = static_cast<int>(sh[2]);
          int W = static_cast<int>(sh[3]);
          std::vector<float> res(N * H * W * C);
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                  int nchw_index = n * C * H * W + c * H * W + h * W + w;
                  int nhwc_index = n * H * W * C + h * W * C + w * C + c;
                  res[nhwc_index] = in[nchw_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(N),
                     static_cast<unsigned long long>(H),
                     static_cast<unsigned long long>(W),
                     static_cast<unsigned long long>(C)});
          output = make_tensor<float>(res, sh1);
          break;
        }
        if (layin_ == NHWC && layout_ == NCHW) {
          int N = static_cast<int>(sh[0]);
          int C = static_cast<int>(sh[3]);
          int H = static_cast<int>(sh[1]);
          int W = static_cast<int>(sh[2]);
          std::vector<float> res(N * C * H * W);
          for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
              for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                  int nhwc_index = n * H * W * C + h * W * C + w * C + c;
                  int nchw_index = n * C * H * W + c * H * W + h * W + w;
                  res[nchw_index] = in[nhwc_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(N),
                     static_cast<unsigned long long>(C),
                     static_cast<unsigned long long>(H),
                     static_cast<unsigned long long>(W)});
          output = make_tensor<float>(res, sh1);
          break;
        }
        output = make_tensor<float>(in, sh);
        break;
      }
      default: {
        throw std::runtime_error("No such type");
      }
    }
  }
};
