#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layers/BatchNormalizationLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class DenseNetPath : public Layer {
 private:
  size_t stride1_;
  size_t pads1_;
  size_t dilations1_;
  std::shared_ptr<Tensor> kernel1_;
  std::shared_ptr<Tensor> bias1_;
  size_t group1_;
  bool useLegacyImpl1_;

  size_t stride2_;
  size_t pads2_;
  size_t dilations2_;
  std::shared_ptr<Tensor> kernel2_;
  std::shared_ptr<Tensor> bias2_;
  size_t group2_;
  bool useLegacyImpl2_;

  Tensor scale_;
  Tensor bias_;
  Tensor mean_;
  Tensor var_;
  float epsilon_;
  float momentum_;
  bool training_mode_;

  template <typename T>
  void batchnormrelu_impl(const Tensor& input, Tensor& output) const;

  void validate_parameters(size_t num_channels) const;

 public:
  DenseNetPath()
      : Layer(kDenseNetPath),
        kernel1_(nullptr),
        bias1_(nullptr),
        kernel2_(nullptr),
        bias2_(nullptr) {
    stride1_ = 0;
    pads1_ = 0;
    dilations1_ = 0;
    group1_ = 0;
    useLegacyImpl1_ = false;
    stride2_ = 0;
    pads2_ = 0;
    dilations2_ = 0;
    group2_ = 0;
    useLegacyImpl2_ = false;
    epsilon_ = 0.0f;
    momentum_ = 0.0f;
    training_mode_ = false;
  }
  explicit DenseNetPath(const std::shared_ptr<BatchNormalizationLayer>& bn,
                        const std::shared_ptr<ConvolutionalLayer>& conv1,
                        const std::shared_ptr<ConvolutionalLayer>& conv2)
      : Layer(kDenseNetPath) {
    auto numerics1 = conv1->getNumericParams();
    auto tensors1 = conv1->getTensorParams();
    stride1_ = numerics1[0];
    pads1_ = numerics1[1];
    dilations1_ = numerics1[2];
    group1_ = numerics1[3];
    kernel1_ = tensors1[0];
    bias1_ = tensors1[1];
    useLegacyImpl1_ = conv1->getLegacyImplBool();

    auto numerics2 = conv2->getNumericParams();
    auto tensors2 = conv2->getTensorParams();
    stride2_ = numerics2[0];
    pads2_ = numerics2[1];
    dilations2_ = numerics2[2];
    group2_ = numerics2[3];
    kernel2_ = tensors2[0];
    bias2_ = tensors2[1];
    useLegacyImpl2_ = conv2->getLegacyImplBool();

    auto numerics3 = bn->getNumericParams();
    auto tensors3 = bn->getTensorParams();
    training_mode_ = bn->getTrainingMode();
    scale_ = tensors3[0];
    bias_ = tensors3[1];
    mean_ = tensors3[2];
    var_ = tensors3[3];
    epsilon_ = numerics3[0];
    momentum_ = numerics3[1];
  }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
           const RuntimeOptions& options) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return *kernel1_; }
#endif
};

}  // namespace it_lab_ai
