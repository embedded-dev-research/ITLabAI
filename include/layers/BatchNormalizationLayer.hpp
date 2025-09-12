#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class BatchNormalizationLayer : public Layer {
 public:
  BatchNormalizationLayer(const Tensor& scale, const Tensor& bias,
                          const Tensor& mean, const Tensor& var,
                          float epsilon = 1e-5f, float momentum = 0.9f,
                          bool training_mode = false)
      : scale_(scale),
        bias_(bias),
        mean_(mean),
        var_(var),
        epsilon_(epsilon),
        momentum_(momentum),
        training_mode_(training_mode) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

  static std::string get_name() { return "BatchNormalizationLayer"; }

  void set_epsilon(float epsilon) { epsilon_ = epsilon; }
  void set_momentum(float momentum) { momentum_ = momentum; }
  void set_training_mode(bool training_mode) { training_mode_ = training_mode; }

 private:
  Tensor scale_;
  Tensor bias_;
  Tensor mean_;
  Tensor var_;
  float epsilon_;
  float momentum_;
  bool training_mode_;

  template <typename T>
  void batchnorm_impl(const Tensor& input, Tensor& output) const;

  void validate_parameters(size_t num_channels) const;
};

}  // namespace it_lab_ai