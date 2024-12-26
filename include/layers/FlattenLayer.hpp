#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class FlattenLayer : public Layer {
 private:
  std::vector<size_t> order;

 public:
  FlattenLayer() : order({0, 1, 2, 3}) {}
  FlattenLayer(const std::vector<size_t>& order) : order(order) {}
  static std::string get_name() { return "Flatten layer"; }
  void run(const Tensor& input, Tensor& output) override;
};

}  // namespace itlab_2023
