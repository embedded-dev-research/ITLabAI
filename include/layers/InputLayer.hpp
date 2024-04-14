#pragma once
#include <algorithm>
#include <cmath>

#include "layers/Layer.hpp"

class InputLayer : public Layer {
 private:
  bool layout_;
  int mean_;
  int std_;

 public:
  InputLayer() = default;
  InputLayer(bool layout, int mean, int std) {
    layout_ = layout;
    mean_ = mean;
    std_ = std;
  }  // layout = NCHW(0), NHWC(1)
  void run(Tensor& input, Tensor& output) {
    switch (input.get_type()) {
      case Type::kInt: {
        std::vector<int> res = *input.as<int>();
        for (size_t i = 0; i < res.size(); ++i) {
          res[i] = static_cast<int>((res[i] - mean_) / std_);
        }
        Shape sh(input.get_shape());
        output = make_tensor<int>(res, sh);
        break;
      }
      case Type::kFloat: {
        std::vector<float> res = *input.as<float>();
        for (size_t i = 0; i < res.size(); ++i) {
          res[i] = static_cast<float>((res[i] - mean_) / std_);
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
