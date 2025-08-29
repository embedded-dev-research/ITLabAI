#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class DropOutLayer : public Layer {
 private:
  double drop_rate_;

 public:
  DropOutLayer() = default;
  DropOutLayer(double drop_rate) { drop_rate_ = drop_rate; }
  static std::string get_name() { return "DropOut layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif
};

}  // namespace it_lab_ai