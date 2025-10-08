#pragma once
#include <algorithm>
#include <cmath>

#include "layers/Layer.hpp"

namespace it_lab_ai {

enum LayInOut : uint8_t {
  kNchw,  // 0
  kNhwc   // 1
};

class InputLayer : public Layer {
 private:
  LayInOut layin_;
  LayInOut layout_;
  int mean_;
  int std_;

 public:
  InputLayer() : Layer(kInput) {
    layin_ = kNchw;
    layout_ = kNchw;
    mean_ = 0;
    std_ = 1;
  }
  InputLayer(LayInOut layin, LayInOut layout, int mean = 0, int std = 1)
      : Layer(kInput) {
    layin_ = layin;
    layout_ = layout;
    mean_ = mean;
    std_ = std;
  }  // layout = kNchw(0), kNhwc(1)
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override {
    if (input.size() != 1) {
      throw std::runtime_error("InputLayer: Input tensors not 1");
    }
    switch (input[0].get_type()) {
      case Type::kInt: {
        std::vector<int> in = *input[0].as<int>();
        if (input[0].get_shape().dims() != 4) {
          throw std::out_of_range(
              "The size of the shape does not match what is needed for the "
              "input layer");
        }
        for (int& re : in) {
          re = (re - mean_) / std_;
        }
        Shape sh(input[0].get_shape());
        if (layin_ == kNchw && layout_ == kNhwc) {
          int n = static_cast<int>(sh[0]);
          int c = static_cast<int>(sh[1]);
          int h = static_cast<int>(sh[2]);
          int w = static_cast<int>(sh[3]);
          if (n < 1 || c < 1 || h < 1 || w < 1) {
            throw std::out_of_range("One of the sizes <= 0");
          }
          std::vector<int> res(n * h * w * c);
          for (int n1 = 0; n1 < n; ++n1) {
            for (int c1 = 0; c1 < c; ++c1) {
              for (int h1 = 0; h1 < h; ++h1) {
                for (int w1 = 0; w1 < w; ++w1) {
                  int nchw_index = n1 * c * h * w + c1 * h * w + h1 * w + w1;
                  int nhwc_index = n1 * h * w * c + h1 * w * c + w1 * c + c1;
                  res[nhwc_index] = in[nchw_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(h),
                     static_cast<unsigned long long>(w),
                     static_cast<unsigned long long>(c)});
          output[0] = make_tensor<int>(res, sh1);
          break;
        }
        if (layin_ == kNhwc && layout_ == kNchw) {
          int n = static_cast<int>(sh[0]);
          int c = static_cast<int>(sh[3]);
          int h = static_cast<int>(sh[1]);
          int w = static_cast<int>(sh[2]);
          if (n < 1 || c < 1 || h < 1 || w < 1) {
            throw std::out_of_range("One of the sizes <= 0");
          }
          std::vector<int> res(n * h * w * c);
          for (int n1 = 0; n1 < n; ++n1) {
            for (int c1 = 0; c1 < c; ++c1) {
              for (int h1 = 0; h1 < h; ++h1) {
                for (int w1 = 0; w1 < w; ++w1) {
                  int nhwc_index = n1 * h * w * c + h1 * w * c + w1 * c + c1;
                  int nchw_index = n1 * c * h * w + c1 * h * w + h1 * w + w1;
                  res[nchw_index] = in[nhwc_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(c),
                     static_cast<unsigned long long>(h),
                     static_cast<unsigned long long>(w)});
          output[0] = make_tensor<int>(res, sh1);
          break;
        }
        output[0] = make_tensor<int>(in, sh);
        break;
      }
      case Type::kFloat: {
        std::vector<float> in = *input[0].as<float>();
        if (input[0].get_shape().dims() != 4) {
          throw std::out_of_range(
              "The size of the shape does not match what is needed for the "
              "input layer");
        }
        for (float& re : in) {
          re = static_cast<float>((re - mean_) / std_);
        }
        Shape sh(input[0].get_shape());
        if (layin_ == kNchw && layout_ == kNhwc) {
          int n = static_cast<int>(sh[0]);
          int c = static_cast<int>(sh[1]);
          int h = static_cast<int>(sh[2]);
          int w = static_cast<int>(sh[3]);
          if (n < 1 || c < 1 || h < 1 || w < 1) {
            throw std::out_of_range("One of the sizes <= 0");
          }
          std::vector<float> res(n * h * w * c);
          for (int n1 = 0; n1 < n; ++n1) {
            for (int c1 = 0; c1 < c; ++c1) {
              for (int h1 = 0; h1 < h; ++h1) {
                for (int w1 = 0; w1 < w; ++w1) {
                  int nchw_index = n1 * c * h * w + c1 * h * w + h1 * w + w1;
                  int nhwc_index = n1 * h * w * c + h1 * w * c + w1 * c + c1;
                  res[nhwc_index] = in[nchw_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(h),
                     static_cast<unsigned long long>(w),
                     static_cast<unsigned long long>(c)});
          output[0] = make_tensor<float>(res, sh1);
          break;
        }
        if (layin_ == kNhwc && layout_ == kNchw) {
          int n = static_cast<int>(sh[0]);
          int c = static_cast<int>(sh[3]);
          int h = static_cast<int>(sh[1]);
          int w = static_cast<int>(sh[2]);
          if (n < 1 || c < 1 || h < 1 || w < 1) {
            throw std::out_of_range("One of the sizes <= 0");
          }
          std::vector<float> res(n * h * w * c);
          for (int n1 = 0; n1 < n; ++n1) {
            for (int c1 = 0; c1 < c; ++c1) {
              for (int h1 = 0; h1 < h; ++h1) {
                for (int w1 = 0; w1 < w; ++w1) {
                  int nhwc_index = n1 * h * w * c + h1 * w * c + w1 * c + c1;
                  int nchw_index = n1 * c * h * w + c1 * h * w + h1 * w + w1;
                  res[nchw_index] = in[nhwc_index];
                }
              }
            }
          }
          Shape sh1({static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(c),
                     static_cast<unsigned long long>(h),
                     static_cast<unsigned long long>(w)});
          output[0] = make_tensor<float>(res, sh1);
          break;
        }
        output[0] = make_tensor<float>(in, sh);
        break;
      }
      default: {
        throw std::runtime_error("No such type");
      }
    }
  }
};

}  // namespace it_lab_ai
