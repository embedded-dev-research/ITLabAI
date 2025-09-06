#pragma once
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ConcatLayer : public Layer {
 public:
  explicit ConcatLayer(int64_t axis = 0) : axis_(axis) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

  static std::string get_name() { return "ConcatLayer"; }

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  int64_t axis_;

  void validate_inputs(const std::vector<Tensor>& inputs) const;
  int64_t normalize_axis(size_t rank) const;
  Shape calculate_output_shape(const std::vector<Tensor>& inputs) const;

  template <typename T>
  void concatenate(const std::vector<Tensor>& inputs, Tensor& output) const;
};

}  // namespace it_lab_ai