#pragma once
#include <optional>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class SplitLayer : public Layer {
 public:
  SplitLayer(int axis, std::vector<int> splits)
      : axis_(axis), splits_(std::move(splits)) {}

  SplitLayer(int axis, int num_outputs)
      : axis_(axis), num_outputs_(num_outputs) {}
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

  static std::string get_name() { return "SplitLayer"; }

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  int axis_;
  std::optional<std::vector<int>> splits_;
  std::optional<int> num_outputs_;

  void validate(const Tensor& input) const;
  int get_normalized_axis(int rank) const;
  template <typename T>
  void split_impl(const Tensor& input, std::vector<Tensor>& outputs) const;
};

}  // namespace it_lab_ai