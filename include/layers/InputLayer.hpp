#pragma once
#include <algorithm>
#include <cmath>

#include "layers/Layer.hpp"

class InputLayer : public Layer {
 private:
  int layout_;
  int mean_;
  int std_;

 public:
  InputLayer() = default;
  InputLayer(int layout, int mean, int std) {
    layout_ = layout;
    mean_ = mean;
    std_ = std;
  }  // layout = NCHW(0), NHWC(1)
  void run(Tensor& input, Tensor& output) const {
    switch (input.get_type()) {
      case Type::kInt: {
        std::vector<int> res = *input.as<int>();
        for (int& re : res) {
          re = static_cast<int>((re - mean_) / std_);
        }
        Shape sh(input.get_shape());
        output = make_tensor<int>(res, sh);
        break;
      }
      case Type::kFloat: {
        std::vector<float> res = *input.as<float>();
        for (float& re : res) {
          re = static_cast<float>((re - mean_) / std_);
        }
        Shape sh(input.get_shape());
        output = make_tensor<float>(res, sh);
        break;
      }
      default: {
        throw std::runtime_error("No such type");
      }
    }
  }
};
