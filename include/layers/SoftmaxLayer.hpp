#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer(int axis = -1) : Layer(kSoftmax), axis_(axis) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

  void set_axis(int axis) { axis_ = axis; }
  int get_axis() const { return axis_; }

 private:
  int axis_;

  template <typename T>
  void softmax_impl(const Tensor& input, Tensor& output) const;

  void softmax_int_impl(const Tensor& input, Tensor& output) const;

  static size_t normalize_axis(const Shape& shape, int axis);
};

}  // namespace it_lab_ai