#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class FlattenLayer : public Layer {
 private:
  std::vector<size_t> order_;

 public:
  FlattenLayer() : order_({0, 1, 2, 3}) {}
  FlattenLayer(const std::vector<size_t>& order) : order_(order) {}
  static std::string get_name() { return "Flatten layer"; }
  void run(const Tensor& input, Tensor& output) override;
};

}  // namespace itlab_2023
