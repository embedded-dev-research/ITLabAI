#pragma once

#include <memory>
#include <string>
#include <vector>

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

class ConvReluLayer : Layer {
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
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
           const RuntimeOptions& options) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return *kernel_; }
#endif
};
}  // namespace it_lab_ai