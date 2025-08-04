#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ReduceLayer : public Layer {
 public:
  enum class Operation { kSum, kMean, kMult, kMax, kMin };

  ReduceLayer(Operation op, int64_t keepdims = 0);
  explicit ReduceLayer(int64_t keepdims = 0)
      : ReduceLayer(Operation::kSum, keepdims) {}
  void run(const Tensor& input, Tensor& output) override;
  void run(const Tensor& input, const Tensor& axes, Tensor& output);

  static std::string get_name() { return "ReduceLayer"; }

 private:
  Operation op_;
  int64_t keepdims_;
  static void normalize_axes(const Shape& input_shape,
                             std::vector<int64_t>& axes);
  Shape calculate_output_shape(const Shape& input_shape,
                               const std::vector<int64_t>& axes) const;

  template <typename T>
  void compute(const Tensor& input, const Shape& output_shape,
               const std::vector<int64_t>& axes, Tensor& output) const;
};

}  // namespace it_lab_ai