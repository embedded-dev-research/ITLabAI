#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class DropOutLayer : public Layer {
 private:
  double drop_rate_;

 public:
  DropOutLayer() = default;
  DropOutLayer(double drop_rate) { drop_rate_ = drop_rate; }
  static std::string get_name() { return "DropOut layer"; }
  void run(const Tensor& input, Tensor& output) override;
};

}  // namespace itlab_2023