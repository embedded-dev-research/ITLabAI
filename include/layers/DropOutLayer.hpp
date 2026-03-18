#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class DropOutLayer : public Layer {
 private:
  double drop_rate_;
  bool training_mode_;

 public:
  explicit DropOutLayer(double drop_rate = 0.0, bool training_mode = false)
      : Layer(kDropout) {
    drop_rate_ = drop_rate;
    training_mode_ = training_mode;
  }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    return Tensor();
  }
#endif
};

}  // namespace it_lab_ai
