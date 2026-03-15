#pragma once
#include <cstdint>
#include <limits>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"
#include "parallel/parallel.hpp"

namespace it_lab_ai {

class ReduceLayer : public Layer {
 public:
  enum class Operation : uint8_t { kSum, kMean, kMult, kMax, kMin };

  explicit ReduceLayer(Operation op, int64_t keepdims = 0,
                       const std::vector<int64_t>& axes = {});

  explicit ReduceLayer(int64_t keepdims = 0,
                       const std::vector<int64_t>& axes = {})
      : ReduceLayer(Operation::kSum, keepdims, axes) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void run(const std::vector<Tensor>& input, std::vector<Tensor>& output,
           const RuntimeOptions& options) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    return Tensor();
  }
#endif

 private:
  Operation op_;
  int64_t keepdims_;
  std::vector<int64_t> axes_;

  static void normalize_axes(const Shape& input_shape,
                             std::vector<int64_t>& axes);
  [[nodiscard]] Shape calculate_output_shape(
      const Shape& input_shape, const std::vector<int64_t>& axes) const;

  template <typename T>
  void compute(const Tensor& input, const Shape& output_shape,
               const std::vector<int64_t>& axes, Tensor& output,
               ParBackend backend = ParBackend::kSeq) const;
};

}  // namespace it_lab_ai
