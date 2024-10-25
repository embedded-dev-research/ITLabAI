#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class FlattenLayer : public Layer {
 public:
  FlattenLayer() = default;
  static std::string get_name() { return "Flatten layer"; }
  void run(const Tensor& input, Tensor& output) override;

};

}  // namespace itlab_2023
