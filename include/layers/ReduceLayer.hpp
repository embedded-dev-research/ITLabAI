#pragma once
#include <cstdint>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ReduceLayer : public Layer {
 public:
  enum class Operation : uint8_t { kSum, kMean, kMult, kMax, kMin };

  ReduceLayer(Operation op, int64_t keepdims = 0,
              const Tensor& axes = make_tensor(std::vector<int>{}));
  explicit ReduceLayer(int64_t keepdims = 0,
                       const Tensor& axes = make_tensor(std::vector<int>{}))
      : ReduceLayer(Operation::kSum, keepdims, axes) {}
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  Operation op_;
  int64_t keepdims_;
  Tensor axes_;
  static void normalize_axes(const Shape& input_shape,
                             std::vector<int64_t>& axes);
  Shape calculate_output_shape(const Shape& input_shape,
                               const std::vector<int64_t>& axes) const;

  template <typename T>
  void compute(const Tensor& input, const Shape& output_shape,
               const std::vector<int64_t>& axes, Tensor& output) const;
};

}  // namespace it_lab_ai