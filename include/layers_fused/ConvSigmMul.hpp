#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layers/BinaryOpLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

template <typename T>
void sigmoid(T& t) {
  if constexpr (std::is_integral_v<T>) {
    auto x_float = static_cast<float>(t);
    float result = 1.0F / (1.0F + std::exp(-x_float));
    t = static_cast<T>(std::round(result));
  }
}

template <typename T>
void sigmoid(Tensor& t) {
  Shape sh = t.get_shape();
  for (size_t i = 0; i < sh.count(); i++) {
    sigmoid<T>((*t.as<T>())[i]);
  }
}

class ConvSigmMulLayer : public Layer {
 private:
  size_t stride_;
  size_t pads_;
  size_t dilations_;
  std::shared_ptr<Tensor> kernel_;
  std::shared_ptr<Tensor> bias_;
  size_t group_;
  bool useLegacyImpl_;

 public:
  ConvSigmMulLayer() : Layer(kConvSigmMul), kernel_(nullptr), bias_(nullptr) {
    stride_ = 0;
    pads_ = 0;
    dilations_ = 0;
  }
  ConvSigmMulLayer(size_t step, size_t pads, size_t dilations,
                   const Tensor& kernel, const Tensor& bias = Tensor(),
                   size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvSigmMul),
        kernel_(std::make_shared<Tensor>(kernel)),
        bias_(std::make_shared<Tensor>(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }
  ConvSigmMulLayer(size_t step, size_t pads, size_t dilations,
                   std::shared_ptr<Tensor> kernel,
                   std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(),
                   size_t group = 1, bool useLegacyImpl = false)
      : Layer(kConvSigmMul),
        kernel_(std::move(kernel)),
        bias_(std::move(bias)) {
    stride_ = step;
    pads_ = pads;
    group_ = group;
    dilations_ = dilations;
    useLegacyImpl_ = useLegacyImpl;
  }
  explicit ConvSigmMulLayer(const std::shared_ptr<ConvolutionalLayer>& conv)
      : Layer(kConvSigmMul) {
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
}  // namespace it_lab_ai
